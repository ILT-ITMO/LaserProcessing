# create_params_presets.py
from pathlib import Path
from itertools import product
from datetime import datetime

from config import Config
from pinn_io import save_params_json

# --- БАЗОВЫЕ ПАРАМЕТРЫ (правьте под себя) ---
BASE_CFG = dict(
    # Геометрия/нормировки
    R_m=1e-3,
    H_m=1e-3,
    w0_m=62e-6,       # ~минимальная ширина канала /2 по e^-2 (оценка)
    Twindow_s=1e-6,   # окно моделирования 1 мкс
    mu_star=100.0,
    T0_C=20.0,

    # Материал (кварц, fused silica)
    rho_kg_m3=2200.0,
    cp_J_kgK=740.0,
    k_W_mK=1.4,

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
        }

        save_params_json(out_path, cfg, extra=extra, pretty=True)
        print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
