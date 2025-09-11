# create_preset_param.py
from __future__ import annotations
from pathlib import Path
from itertools import product
from datetime import datetime
import math

from config import Config
from pinn_io import save_params_json

# --- БАЗОВЫЕ ПАРАМЕТРЫ (правьте под себя) ---
BASE_CFG = dict(
    # Геометрия/нормировки
    R_m=300e-6,
    H_m=100e-6,
    w0_m=162e-6,
    Twindow_s=1e-6,
    mu_star=1.0,
    T0_C=20.0,

    # Материал (кварц)
    rho_kg_m3=2200.0,
    cp_J_kgK=740.0,
    k_W_mK=1.4,

    # Нормировка температуры
    temp_scaling_mode="A",   # "A" | "B" | "C"
    eta_abs=1.0,
    U_target=0.9,
    kappa_r=0.5, kappa_z=0.5,

    # Импульсные поля (опционально; можно оставить None)
    pulse_duration_s=1e-6,   # FWHM, s
    rep_rate_Hz=8e3,        # Hz
    pulse_count=3,        # N импульсов
    pulses_t0_s=1e-6,         # стартовая задержка
    E_pulse_J=None,          # J
    P_avg_W=None,            # если задана — приоритетнее P_W
    # P_W зададим перебором ниже

    # Сэмплинг/обучение
    N_coll=20000, N_ic=4096, N_axis=2048, N_wall=2048, N_zbc=1024,
    widths=(64, 64, 64, 64), lr=1e-3,
    w_pde=1.0, w_data=1.0, w_ic=10.0, w_bc=10.0,
    seed=42, device="cpu",
)

POWERS_W = [3.3]     # набор средних мощностей (если P_avg_W=None, то пойдёт в P_W)
SCAN_SPEED_MM_S = [40]

OUT_DIR = Path("presets_params")
OUT_DIR.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


def make_cfg(power_w: float) -> Config:
    """
    Если в BASE_CFG задана P_avg_W — используем её как среднюю мощность.
    Иначе — кладём 'наследуемое' P_W = power_w.
    """
    if BASE_CFG.get("P_avg_W", None) is not None:
        return Config(P_W=power_w, **BASE_CFG)  # P_W сохранится, но P_avg_W будет иметь приоритет
    else:
        return Config(P_W=power_w, **BASE_CFG)


def _recompute_time_window_if_needed(cfg: Config) -> None:
    """
    Если заданы rep_rate_Hz и pulse_count, пересчитать Twindow_s так,
    чтобы окно полностью вмещало последовательность:
        Twindow >= pulses_t0_s + (pulse_count - 1)/rep_rate_Hz
    """
    if (cfg.rep_rate_Hz is None) or (cfg.pulse_count is None):
        return
    if (cfg.rep_rate_Hz <= 0) or (cfg.pulse_count <= 0):
        return

    need = float(cfg.pulses_t0_s) + (float(cfg.pulse_count) - 1.0) / float(cfg.rep_rate_Hz)
    if need > cfg.Twindow_s:
        cfg.Twindow_s = need


def compute_deltaT_scale_K(cfg: Config) -> float:
    """
    ΔT_scale (K per unit U) согласован с безразмерной формой PDE:
      U = (T - T0) / ΔT_scale,
      S = (Twindow / ΔT_scale) * Q_phys / (rho * cp).

    Режимы:
      A: "unit source" — S_peak ≈ 1
      B: "single pulse" — вклад одного импульса ~ U_target
      C: "window energy / V_eff"
    """
    rho = float(cfg.rho_kg_m3)
    cp  = float(cfg.cp_J_kgK)
    Tw  = float(cfg.Twindow_s)
    w0  = float(cfg.w0_m)
    H   = float(cfg.H_m)
    eta = float(getattr(cfg, "eta_abs", 1.0))

    # Средняя мощность для расчёта (если нужна)
    P_avg = float(cfg.P_avg_W) if (cfg.P_avg_W is not None) else float(cfg.P_W)

    alpha = float(cfg.mu_star) / H  # μ_a [1/m] = mu_star / H

    # Оценка пикового объёмного источника для режима A/B:
    Q_peak = eta * P_avg * (2.0 / (math.pi * w0**2)) * alpha

    mode = (getattr(cfg, "temp_scaling_mode", "A") or "A").upper()
    if mode == "A":
        return float((Tw * Q_peak) / (rho * cp))

    if mode == "B":
        t_p = float(getattr(cfg, "pulse_duration_s", Tw))  # если не задано — Tw
        U_t = float(getattr(cfg, "U_target", 0.9))
        U_t = max(1e-6, U_t)
        return float((Q_peak * t_p) / (rho * cp) / U_t)

    if mode == "C":
        if (cfg.E_pulse_J is not None) and (cfg.rep_rate_Hz is not None):
            n_p = Tw * float(cfg.rep_rate_Hz)
            E_abs_win = eta * float(cfg.E_pulse_J) * n_p
        else:
            E_abs_win = eta * P_avg * Tw

        R_eff = float(cfg.kappa_r) * w0
        H_eff = float(cfg.kappa_z) * min(H, 1.0 / alpha)
        V_eff = math.pi * R_eff**2 * H_eff
        V_eff = max(V_eff, 1e-18)
        return float(E_abs_win / (rho * cp * V_eff))

    raise ValueError(f"Unknown temp_scaling_mode: {mode}")


def main():
    for p_w, v_mm_s in product(POWERS_W, SCAN_SPEED_MM_S):
        cfg = make_cfg(p_w)

        # 1) Пересчёт окна, если задана гребёнка импульсов
        _recompute_time_window_if_needed(cfg)

        # 2) Имя файла
        p_str = str(p_w).replace(".", "p")
        fname = f"pinn_params_P{p_str}W_V{v_mm_s}mms_{timestamp}.json"
        out_path = OUT_DIR / fname

        # 3) Доп.поля в JSON
        extra = {
            "SCAN_SPEED_MM_S": v_mm_s,
            "NOTE": "Preset grid (power, scan_speed) with pulse-window consistency",
            "DELTA_T_SCALE_K": compute_deltaT_scale_K(cfg),
        }

        # Прокидываем импульсные параметры в JSON для source.py
        if cfg.pulse_duration_s is not None:
            extra["PULSE_FWHM_S"] = float(cfg.pulse_duration_s)
        if cfg.rep_rate_Hz is not None:
            extra["PULSE_REP_HZ"] = float(cfg.rep_rate_Hz)
        if cfg.pulse_count is not None:
            extra["PULSE_COUNT"] = int(cfg.pulse_count)
        if getattr(cfg, "pulses_t0_s", 0.0) != 0.0:
            extra["PULSES_T0_S"] = float(cfg.pulses_t0_s)

        if cfg.P_avg_W is not None:
            extra["P_AVG_W"] = float(cfg.P_avg_W)

        # 4) Сохраняем JSON
        save_params_json(out_path, cfg, extra=extra, pretty=True)
        print(f"Saved: {out_path} (Twindow_s={cfg.Twindow_s:.6e} s)")

if __name__ == "__main__":
    main()
