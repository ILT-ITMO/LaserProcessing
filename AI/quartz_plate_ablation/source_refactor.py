from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

import json
import math
import torch
import torch.nn as nn

from AI.measure_isotherm_width import delta_T_scale_K
from config import Config                # dataclass-конфигурация
from pinn_io import load_params_json     # загрузка строго в Config


def load_cfg_and_extras(json_path: Path) -> Tuple[Config, Dict[str, Any]]:
    """
    Загружает Config и возвращает dict со ВСЕМИ полями JSON.
    Нужно, чтобы вытащить доп. поля (например, SCAN_SPEED_MM_S),
    которые не входят в Config.
    """
    json_path = Path(json_path)
    cfg = load_params_json(json_path)
    data = json.loads(json_path.read_text(encoding="utf-8"))
    return cfg, data

@dataclass
class SourcePhys:
    # Геометрия/материал/нормировки
    w0_m: float
    R_m: float
    H_m: float
    Twindow_s: float
    mu_star: float
    rho_kg_m3: float
    cp_J_kgK: float
    deltaT_scale: float
    # Временные параметры
    pulse_fwhm_s: Optional[float] = None
    rep_rate_Hz: Optional[float] = None
    pulse_count: Optional[int] = None
    pulses_t0_s: float = 0.0
    P_avg_W: Optional[float] = None
    P_peak_W: Optional[float] = None
    # Прочее
    @property
    def mu_abs_m_inv(self) -> float:
        # Function return normolazed mu coefficient μ_a = mu_star / H
        return float(self.mu_star / self.H_m)

    @property
    def gaussian_temporal_area_coeff(self) -> float:
        #Function returns temporal integrall of normalized gaussian pulse; To obtain pulse energy one have to multiply gaussian_temporal_area_coeff by P_peak
        # ∫ exp(-4 ln2 * t^2 / FWHM^2) dt = FWHM * sqrt(pi/(4 ln 2))
        if self.pulse_fwhm_s is None:
            return 1.0
        return self.pulse_fwhm_s * math.sqrt(math.pi / (4.0 * math.log(2.0)))


class OpticalSource(nn.Module):
    """
    S(ρ, ζ, τ) — безразмерный источник для уравнения на U(ρ, ζ, τ)
    c временной огибающей как гребёнка гауссовых импульсов.
    """
    def __init__(self, params: SourcePhys, device: Optional[str] = None):
        super().__init__()
        self.p = params
        # предрасчёт временных центров импульсов (в секундах)
        self._pulse_centers_s: List[float] = self._compute_pulse_centers()
        self.register_buffer("one", torch.tensor(1.0))
        if device is not None:
            self.to(device)
        # Перевод в пиковую мощность
        self._infer_peak_power_from_average()

    # --------- служебные расчёты времени/мощности ----------

    def _compute_pulse_centers(self) -> List[float]:
        p = self.p
        t0 = float(p.pulses_t0_s or 0.0)
        Tw = float(p.Twindow_s)
        # Приоритет: заданный pulse_count; иначе из rep_rate*window; иначе одиночный
        if p.rep_rate_Hz and p.rep_rate_Hz > 0.0:
            if p.pulse_count is None:
                count = int(max(0, math.floor(p.rep_rate_Hz * Tw)))
            else:
                count = int(max(0, p.pulse_count))
            centers = [t0 + k / p.rep_rate_Hz for k in range(count)]
        else:
            # без частоты: одиночный импульс, центр в середине окна
            centers = [t0 if (0.0 <= t0 <= Tw) else 0.5 * Tw]

        # Оставим только те, что попадают в окно
        return [t for t in centers if (0.0 <= t <= Tw)]

    def _infer_peak_power_from_average(self) -> None:
        """
        Если P_peak_W не задана, но есть средняя мощность (P_avg_W или P_W из cfg),
        и заданы rep_rate_Hz и pulse_fwhm_s, то восстановим пиковую мощность.
        """
        p = self.p
        if p.P_peak_W is not None:
            return
        if (p.P_avg_W is None) or (p.pulse_fwhm_s is None) or (p.rep_rate_Hz is None) or (p.rep_rate_Hz <= 0.0):
            return
        area = p.gaussian_area_coeff  # FWHM*sqrt(pi/(4 ln2))
        if area <= 0.0:
            return
        # P_avg = P_peak * area * f_rep  =>  P_peak = P_avg / (area * f_rep)
        P_peak = float(p.P_avg_W) / (area * float(p.rep_rate_Hz))
        self.p.P_peak_W = max(P_peak, 0.0)

    # --------- временная огибающая (торч) ----------

    def temporal_envelope(self, tau: torch.Tensor) -> torch.Tensor:
        """
        Возвращает Σ_k exp(-4 ln2 * (t - t_k)^2 / FWHM^2) в момент времени t = tau*Twindow.
        Если FWHM или центры не определены — возвращает единицу (CW).
        """
        if (self.p.pulse_fwhm_s is None) or (len(self._pulse_centers_s) == 0):
            return torch.ones_like(tau)

        t = tau * self.p.Twindow_s  # сек
        fwhm = float(self.p.pulse_fwhm_s)
        c = -4.0 * math.log(2.0) / (fwhm * fwhm)

        # Сумма гауссов по центрам. Для численной устойчивости — складываем в торче.
        env = torch.zeros_like(tau)
        for tk in self._pulse_centers_s:
            env = env + torch.exp(c * (t - tk) ** 2)
        return env

    @property
    def deltaT_scale_K(self) -> float:
        """Удобный доступ к масштабу ΔT (K на единицу U), проброшенному из пресета."""
        return float(self.p.deltaT_scale_K)

    # --- физический q(r,z,t) в Вт/м^3 ---
    def q_W_m3(self, rho: torch.Tensor, zeta: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        """
        q = μ_a * I_surf(r,t) * exp(-μ_a * z),
        I_surf(r,t) = (2 P(t) / (π w0^2)) * exp(-2 r^2 / w0^2)
        """
        r = rho * self.p.R_m   # м
        z = zeta * self.p.H_m  # м

        # Вычислим мгновенную мощность P(t)
        env = self.temporal_envelope(tau)  # сумма гауссов без масштаба
        # Масштаб мощности: используем пиковую мощность; если её нет — fallback к P_avg или 0.
        if self.p.P_peak_W is not None:
            P_t = float(self.p.P_peak_W) * env
        elif self.p.P_avg_W is not None:
            # если нет rep_rate/FWHM, env≈1 => трактуем как CW со средней мощностью
            P_t = float(self.p.P_avg_W) * env
        else:
            # финальный запасной вариант — CW с cfg.P_W (может быть 0)
            P_t = float(getattr(self.p, "P_avg_W", 0.0)) * env  # обычно недостижимо

        # I(r,t) = 2P(t)/(π w0^2) * exp(-2 r^2 / w0^2)
        I0 = 2.0 / (math.pi * (self.p.w0_m ** 2))
        I_r_t = I0 * P_t * torch.exp(-2.0 * (r ** 2) / (self.p.w0_m ** 2))

        mu_a = self.p.mu_abs_m_inv
        q = mu_a * I_r_t * torch.exp(-mu_a * z)
        return q  # Вт/м^3

    # --- безразмерный S(ρ,ζ,τ) ---
    def forward(self, rho: torch.Tensor, zeta: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        """
        S = (Twindow_s / (ρ cp ΔT_scale)) * q_W_m3
        """
        q = self.q_W_m3(rho, zeta, tau)
        scale = self.p.Twindow_s / (self.p.rho_kg_m3 * self.p.cp_J_kgK * self.p.deltaT_scale_K)
        return q * scale

    def build_optical_source(
            params_json_path: str | Path,
    ) -> OpticalSource:
        """
        Создаёт объект источника, используя пресетный JSON.
        Читает DELTA_T_SCALE_K, рассчитанный ранее в create_preset_param.py.
        Параметры импульсной последовательности берутся из:
          - Config: pulse_duration_s (FWHM), rep_rate_Hz
          - Extras(JSON): PULSE_FWHM_S, PULSE_REP_HZ, PULSE_COUNT, PULSES_T0_S,
                          PULSE_PEAK_W, P_AVG_W
        При необходимости можно явно переопределить deltaT_scale_override.
        """
        cfg, data = load_cfg_and_extras(Path(params_json_path))

        # Прочтём временные параметры с приоритетом: extras -> cfg
        pulse_fwhm_s = cfg.pulse_duration_s
        rep_rate_Hz = cfg.rep_rate_Hz
        pulse_count = cfg.pulse_count
        pulses_t0_s = cfg.pulses_t0_s
        P_avg_W = cfg.P_avg_W
        device = cfg.device
        deltaT_scale = cfg.deltaT_scale
        src = OpticalSource(
            SourcePhys(
                # Геометрия/материал/нормировки
                w0_m=cfg.w0_m,
                R_m=cfg.R_m,
                H_m=cfg.H_m,
                Twindow_s=cfg.Twindow_s,
                mu_star=cfg.mu_star,
                rho_kg_m3=cfg.rho_kg_m3,
                cp_J_kgK=cfg.cp_J_kgK,
                deltaT_scale=deltaT_scale,
                # Временные параметры
                pulse_fwhm_s=float(pulse_fwhm_s) if pulse_fwhm_s is not None else None,
                rep_rate_Hz=float(rep_rate_Hz) if rep_rate_Hz is not None else None,
                pulse_count=int(pulse_count) if pulse_count is not None else None,
                pulses_t0_s=float(pulses_t0_s),
                P_avg_W=float(P_avg_W),
            ),
            device=device or getattr(cfg, "device", None),
        )
        return src


PARAMS_JSON = Path("presets_params/pinn_params_P3p3W_V40mms_20250911_164958.json")
data,_ = load_cfg_and_extras(PARAMS_JSON)
print(data)