# create_params_presets.py
from pathlib import Path
from itertools import product
from datetime import datetime

from config import Config
from pinn_io import save_params_json
import math
# --- БАЗОВЫЕ ПАРАМЕТРЫ (правьте под себя) ---
BASE_CFG = dict(
    # Геометрия/нормировки
    R_m=300e-6,
    H_m=100e-6,
    w0_m=162e-6,       # ~минимальная ширина канала /2 по e^-2 (оценка)
    Twindow_s=1e-6,   # окно моделирования 1 мкс
    mu_star=10.0,
    T0_C=20.0,

    # Материал (кварц, fused silica)
    rho_kg_m3=2200.0,
    cp_J_kgK=740.0,
    k_W_mK=1.4,

    # Нормировка температуры: выберите "A" | "B" | "C"
    temp_scaling_mode="A",
    eta_abs=1.0,
    U_target=0.9,
    # pulse_duration_s=None,
    # rep_rate_Hz=None, E_pulse_J=None,
    # P_avg_W=None,
    kappa_r=0.5, kappa_z=0.5,

    # Сэмплинг/обучение
    N_coll=20000, N_ic=4096, N_axis=2048, N_wall=2048, N_zbc=1024,
    widths=(64,64,64,64), lr=1e-3,
    w_pde=1.0, w_data=1.0, w_ic=10.0, w_bc=10.0,
    seed=42, device="cpu",
)

POWERS_W = [3.3, 11.1, 13.6]         # Вт
SCAN_SPEED_MM_S = [40, 50, 60]       # мм/с

OUT_DIR = Path("presets_params")
OUT_DIR.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

def make_cfg(power_w: float) -> Config:
    return Config(P_W=power_w, **BASE_CFG)


def compute_deltaT_scale_K(cfg: Config) -> float:
    """
    ΔT_scale (K per unit U) строго согласован с безразмерной формой PDE:
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
    P_W = float(cfg.P_W)
    w0  = float(cfg.w0_m)
    H   = float(cfg.H_m)
    eta = float(getattr(cfg, "eta_abs", 1.0))

    # Коэф. поглощения по глубине из безразмерного mu_star: α = mu_star / H
    alpha = float(cfg.mu_star) / H  # [1/m]

    # Пиковая объемная мощность (Beer–Lambert для гаусса)
    # Q_peak [W/m^3] ≈ η * P_peak * (2/(π w0^2)) * α
    Q_peak = eta * P_W * (2.0 / (math.pi * w0**2)) * alpha

    mode = (getattr(cfg, "temp_scaling_mode", "A") or "A").upper()
    if mode == "A":
    # S_peak ≈ 1  => ΔT_scale = (Tw * Q_peak) / (rho * cp)
        return float((Tw * Q_peak) / (rho * cp))

    if mode == "B":
        t_p = float(getattr(cfg, "pulse_duration_s", Tw))  # если не задано — берем Tw
        U_t = float(getattr(cfg, "U_target", 0.9))
        U_t = max(1e-6, U_t)
        # вклад одного импульса: ΔT ≈ (Q_peak * t_p)/(rho*cp)
        return float((Q_peak * t_p) / (rho * cp) / U_t)

    if mode == "C":
        # E_abs за окно Tw
        if getattr(cfg, "E_pulse_J", None) is not None and getattr(cfg, "rep_rate_Hz", None) is not None:
           n_p = Tw * float(cfg.rep_rate_Hz)
           E_abs_win = eta * float(cfg.E_pulse_J) * n_p
        else:
            P_avg = float(getattr(cfg, "P_avg_W", P_W))
            E_abs_win = eta * P_avg * Tw
        R_eff = float(cfg.kappa_r) * w0
        H_eff = float(cfg.kappa_z) * min(H, 1.0/alpha)
        V_eff = math.pi * R_eff**2 * H_eff
        V_eff = max(V_eff, 1e-18)
        return float(E_abs_win / (rho * cp * V_eff))

    raise ValueError(f"Unknown temp_scaling_mode: {mode}")

def main():
    for p_w, v_mm_s in product(POWERS_W, SCAN_SPEED_MM_S):
        cfg = make_cfg(p_w)
        p_str = str(p_w).replace(".", "p")
        # имя файла: params_P{power}_V{speed}.json
        fname = f"pinn_params_P{p_str}W_V{v_mm_s}mms_{timestamp}.json"
        out_path = OUT_DIR / fname

        # дополнительные поля в JSON (не в Config)
        extra = {
            "SCAN_SPEED_MM_S": v_mm_s,
            "NOTE": "MF chip; CO2 laser; preset grid of (power, scan_speed)",
            "DELTA_T_SCALE_K": compute_deltaT_scale_K(cfg),
        }

        save_params_json(out_path, cfg, extra=extra, pretty=True)
        print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
