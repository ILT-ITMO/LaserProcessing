# source.py
"""
Оптический источник для PINN с цилиндрической симметрией.
Читает параметры из пресетного JSON и возвращает безразмерный источник S(ρ, ζ, τ).

Нормировки:
  r = ρ * R_m, z = ζ * H_m, t = τ * Twindow_s
  T = T0_C + ΔT_scale * U  (U — предсказывает PINN)

Тогда безразмерный источник в уравнении для U:
  S(ρ,ζ,τ) = (Twindow_s / (rho*cp*ΔT_scale)) * q_W_m3(r,z,t)

Где q_W_m3 — объёмная мощность поглощения Гауссова пучка
с экспоненциальным затуханием по глубине (Beer–Lambert):
  q(r,z,t) = μ_a * I_surf(r,t) * exp(-μu_star* z),  μ_a = mu_star / H_m
  I_surf(r,t) = (2 P(t) / (π w0^2)) * exp(-2 r^2 / w0^2)

Временной профиль — гребёнка гауссовых импульсов:
  P(t) = P_peak * Σ_k exp(-4 ln 2 * (t - t_k)^2 / FWHM^2),   t_k = t0 + k/f_rep
Если задана только средняя мощность P_avg, то:
  P_peak = P_avg / (f_rep * FWHM * sqrt(π / (4 ln 2))).
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

import json
import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from config import Config                # dataclass-конфигурация
from pinn_io import load_params_json     # загрузка строго в Config

# ---------- Чтение параметров и "экстры" из JSON ----------

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


# source.py
import math
from typing import Optional
from config import Config  # тип для подсказок; можно убрать при необходимости

_FWHM_TO_SIGMA = 1.0 / (2.0 * math.sqrt(2.0 * math.log(2.0)))  # ≈ 0.42466

def recompute_time_window_if_needed(cfg: "Config") -> None:
    """
    Гарантирует, что Twindow_s достаточно велико, чтобы поместились ВСЕ импульсы
    при заданных pulse_count / rep_rate_Hz / pulses_t0_s / pulse_duration_s.

    Мутирует cfg.Twindow_s по месту.
    """
    if not (getattr(cfg, "Pulsed", False)
            and cfg.pulse_count
            and cfg.rep_rate_Hz
            and cfg.pulse_duration_s is not None):
        return  # CW-режим или неполные данные — ничего не делаем

    Np = int(cfg.pulse_count)
    rep = float(cfg.rep_rate_Hz)
    t0 = float(cfg.pulses_t0_s)
    fwhm = float(cfg.pulse_duration_s)

    # Длительность гребёнки по центрам: от первого до последнего центра
    train_span = 0.0 if Np <= 1 else (Np - 1) / rep
    last_center_time = t0 + train_span

    # Подушка безопасности: ±3σ от центра последнего импульса (и небольшой запас слева)
    sigma_t = fwhm * _FWHM_TO_SIGMA
    safety_pad = 3.0 * sigma_t

    needed_Twindow = max(
        cfg.Twindow_s,                      # текущее окно (если уже больше — не уменьшаем)
        last_center_time + safety_pad       # чтобы «хвост» последнего импульса влез
    )

    # Если первый импульс стартует не в нуле, можно учесть и левый хвост
    # (опционально, но можно добавить небольшой глобальный запас)
    left_pad = max(0.0, 3.0 * sigma_t - t0)
    needed_Twindow += max(0.0, left_pad)

    if needed_Twindow > cfg.Twindow_s:
        cfg.Twindow_s = needed_Twindow




# ---------- Параметры источника ----------

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
    def mu_a(self) -> float:
        # μ_a = mu_star / H
        return float(self.mu_star / self.H_m)

    @property
    def gaussian_area_coeff(self) -> float:
        # ∫ exp(-4 ln2 * t^2 / FWHM^2) dt = FWHM * sqrt(pi/(4 ln 2))
        if self.pulse_fwhm_s is None:
            return 1.0
        return self.pulse_fwhm_s * math.sqrt(math.pi / (4.0 * math.log(2.0)))


# ---------- Источник ----------
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
    def deltaT_scale(self) -> float:
        """Удобный доступ к масштабу ΔT (K на единицу U), проброшенному из пресета."""
        return float(self.p.deltaT_scale)

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
            # финальный запасной вариант — CW с cfg.P_avg_W (может быть 0)
            P_t = float(getattr(self.p, "P_avg_W", 0.0)) * env  # обычно недостижимо

        # I(r,t) = 2P(t)/(π w0^2) * exp(-2 r^2 / w0^2)
        # In case when w0_m is a 1/e2 intensity raduis
        I0 = 2.0 / (math.pi * (self.p.w0_m ** 2))
        I_r_t = I0 * P_t * torch.exp(-2.0 * (r ** 2) / (self.p.w0_m ** 2))
        mu_a = self.p.mu_a
        q = mu_a * I_r_t * torch.exp(-self.p.mu_star * z)
        return q  # Вт/м^3

    # --- безразмерный S(ρ,ζ,τ) ---
    def forward(self, rho: torch.Tensor, zeta: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        """
        S = (Twindow_s / (ρ cp ΔT_scale)) * q_W_m3
        """
        q = self.q_W_m3(rho, zeta, tau)
        scale = self.p.Twindow_s / (self.p.rho_kg_m3 * self.p.cp_J_kgK * self.p.deltaT_scale)
        return q * scale


# ---------- Фабрика ----------

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
            deltaT_scale = deltaT_scale,

            # Временные параметры
            pulse_fwhm_s=float(pulse_fwhm_s) if pulse_fwhm_s is not None else None,
            rep_rate_Hz=float(rep_rate_Hz) if rep_rate_Hz is not None else None,
            pulse_count=int(pulse_count) if pulse_count is not None else None,
            pulses_t0_s=float(pulses_t0_s),
            P_avg_W= float(cfg.P_avg_W),
        ),
        device=device or getattr(cfg, "device", None),
    )
    return src


# ---------- Утилита-сэмплер (опционально) ----------

@torch.no_grad()
def sample_collocation_from_cfg(cfg: Config, *, device: Optional[str] = None, N: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Простейший равномерный сэмплер (ρ, ζ, τ) ~ U(0,1), согласованный с cfg.
    Если N не указан — берём cfg.N_coll.
    Это лишь утилита; можно использовать ваш внешний скрипт сэмплинга.
    """
    device = device or getattr(cfg, "device", "cpu")
    N = int(N or cfg.N_coll)

    rho = torch.rand(N, 1, device=device)
    zeta = torch.rand(N, 1, device=device)
    tau = torch.rand(N, 1, device=device)
    return rho, zeta, tau


# ---------- Визуализация источника (опционально) ----------

def plot_source_2d(
    source: OpticalSource,
    *,
    tau_value: float = 0.5,
    Nr: int = 200,
    Nz: int = 200,
    device: Optional[str] = None,
    overlay_points: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    title: Optional[str] = None,
    show: bool = True,
    savepath: Optional[str | Path] = None,
):
    """
    Рисует карту S(ρ, ζ) при фиксированном τ (в безразмерных координатах).
    """

    device = device or next(source.parameters(), torch.tensor(0.)).device
    rho_lin = torch.linspace(0.0, 1.0, Nr, device=device)
    zeta_lin = torch.linspace(0.0, 1.0, Nz, device=device)
    RHO, ZETA = torch.meshgrid(rho_lin, zeta_lin, indexing="ij")
    TAU = torch.full_like(RHO, float(tau_value))

    with torch.no_grad():
        S = source(RHO, ZETA, TAU).detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(6.2, 4.8), dpi=120)
    im = ax.imshow(S.T, origin="lower", extent=[0, 1, 0, 1], aspect="auto")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("S (безразмерный)")
    ax.set_xlabel("ρ")
    ax.set_ylabel("ζ")
    ax.set_title(title or f"S(ρ, ζ) при τ = {tau_value:.2f}")

    if overlay_points is not None:
        rho_pts, zeta_pts = overlay_points
        rp = rho_pts.detach().flatten().cpu().numpy()
        zp = zeta_pts.detach().flatten().cpu().numpy()
        ax.plot(rp, zp, "o", ms=2.5, mfc="none")

    fig.tight_layout()
    if savepath is not None:
        fig.savefig(savepath, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_source_2d_physical(
    source: OpticalSource,
    *,
    tau_value: float = 0.5,
    Nr: int = 200,
    Nz: int = 200,
    r_unit: str = "mm",     # "m" | "mm" | "um"
    z_unit: str = "um",     # "m" | "mm" | "um"
    quantity: str = "q",    # "q" -> Вт/м^3, "S" -> безразмерный
    device: Optional[str] = None,
    overlay_points: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # (rho_pts, zeta_pts) в [0,1]
    title: Optional[str] = None,
    show: bool = True,
    savepath: Optional[str | Path] = None,
):
    """
    Рисует 2D-карту источника в ФИЗИЧЕСКИХ координатах r–z.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # факторы перевода осей
    unit_factor = {"m": 1.0, "mm": 1e3, "um": 1e6}
    if r_unit not in unit_factor or z_unit not in unit_factor:
        raise ValueError("r_unit/z_unit must be one of: 'm', 'mm', 'um'")

    p = source.p  # SourcePhys
    r_scale = unit_factor[r_unit]
    z_scale = unit_factor[z_unit]

    device = device or next(source.parameters(), torch.tensor(0.)).device
    # нормированные сетки
    rho_lin = torch.linspace(0.0, 1.0, Nr, device=device)
    zeta_lin = torch.linspace(0.0, 1.0, Nz, device=device)
    RHO, ZETA = torch.meshgrid(rho_lin, zeta_lin, indexing="ij")
    TAU = torch.full_like(RHO, float(tau_value))

    # считаем поле
    with torch.no_grad():
        if quantity.lower() == "q":
            VAL = source.q_W_m3(RHO, ZETA, TAU)              # Вт/м^3
            cbar_label = "q (Вт/м³)"
        elif quantity.lower() == "s":
            VAL = source(RHO, ZETA, TAU)                     # безразмерный
            cbar_label = "S (безразмерный)"
        else:
            raise ValueError("quantity must be 'q' or 'S'")

        VAL = VAL.detach().cpu().numpy()

    # физические пределы осей в выбранных единицах
    r_max = p.R_m * r_scale
    z_max = p.H_m * z_scale

    fig, ax = plt.subplots(figsize=(6.6, 4.8), dpi=120)
    im = ax.imshow(
        VAL.T,
        origin="lower",
        extent=[0.0, r_max, 0.0, z_max],  # r, z
        aspect="auto",
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)
    ax.set_xlabel(f"r, {r_unit}")
    ax.set_ylabel(f"z, {z_unit}")
    t_us = float(tau_value) * source.p.Twindow_s * 1e6  # реальное время, мкс
    name = "q" if quantity.lower() == "q" else "S"
    ax.set_title(title or f"{name}(r, z) при τ = {tau_value:.2f}  (t = {t_us:.2f} µs)")

    # наложение точек (ρ, ζ) -> (r, z) в выбранных единицах
    if overlay_points is not None:
        rho_pts, zeta_pts = overlay_points
        r_pts = (rho_pts.detach().flatten().cpu().numpy() * p.R_m) * r_scale
        z_pts = (zeta_pts.detach().flatten().cpu().numpy() * p.H_m) * z_scale
        ax.plot(r_pts, z_pts, "o", ms=2.5, mfc="none")

    fig.tight_layout()
    if savepath is not None:
        fig.savefig(savepath, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

def plot_temporal_envelope(
    source: OpticalSource,
    *,
    Nt: int = 2000,
    normalize: bool = True,
    t_unit: str = "us",         # "s" | "ms" | "us" | "ns"
    show: bool = True,
    savepath: Optional[str | Path] = None,
    title: Optional[str] = None,
):
    """
    Рисует временную огибающую мощности P(t) для заданного источника.
    Если normalize=True — график строится в долях от пика (максимум = 1).
    Иначе — в ваттах, если известна P_peak_W или P_avg_W.

    Параметры:
      Nt       — число точек по времени в окне [0, Twindow_s]
      t_unit   — единицы времени для оси: 's'|'ms'|'us'|'ns'
    """
    import numpy as np
    import matplotlib.pyplot as plt

    unit_scale = {"s": 1.0, "ms": 1e3, "us": 1e6, "ns": 1e9}
    if t_unit not in unit_scale:
        raise ValueError("t_unit must be one of: 's', 'ms', 'us', 'ns'")

    Tw = float(source.p.Twindow_s)
    t = np.linspace(0.0, Tw, int(Nt))
    tau = torch.from_numpy(t / Tw).float()

    with torch.no_grad():
        env = source.temporal_envelope(tau).cpu().numpy()

    # Масштабируем к реальной мощности, если есть данные
    if source.p.P_peak_W is not None:
        P = env * float(source.p.P_peak_W)
        ylabel = "P(t), W"
    elif source.p.P_avg_W is not None:
        # Если нет rep_rate/FWHM, env≈1 и график будет просто константой P_avg
        P = env * float(source.p.P_avg_W)
        ylabel = "P(t), W"
    else:
        # Нет информации о ваттах — показываем безразмерную огибающую
        P = env
        ylabel = "Envelope (arb.)"

    if normalize:
        m = np.max(P)
        if m > 0:
            P = P / m
        ylabel = "P(t) / max"

    t_plot = t * unit_scale[t_unit]

    plt.figure(figsize=(6.4, 3.2), dpi=120)
    plt.plot(t_plot, P)
    plt.xlabel(f"t, {t_unit}")
    plt.ylabel(ylabel)
    plt.title(title or "Временная огибающая импульсов")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()

if __name__ == "__main__":
    # Пример самопроверки: замените путь на актуальный ваш JSON
    PARAMS_JSON = Path("presets_params/pinn_params_P3p3W_V40mms_20250917_132437.json")
    source = build_optical_source(PARAMS_JSON)

    # Пример: наложим коллокационные точки
    cfg, _ = load_cfg_and_extras(PARAMS_JSON)
    rho_pts, zeta_pts, tau_pts = sample_collocation_from_cfg(cfg, N=min(1000, getattr(cfg, "N_coll", 1000)))
    plot_source_2d_physical(
        source,
        tau_value=0.5,
        overlay_points=(rho_pts, zeta_pts),
        savepath=None,  # например: "source_map.png"
    )

    # # Проверка: огибающая во времени (в микросекундах), нормированная
    # plot_temporal_envelope(
    #     source,
    #     Nt=4000,
    #     normalize=True,
    #     t_unit="us",
    #     show=True,
    #     savepath=None,  # например: "temporal_envelope.png"
    # )
