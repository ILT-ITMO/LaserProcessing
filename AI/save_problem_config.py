# -*- coding: utf-8 -*-
# save_problem_config.py
# Создает JSON с параметрами задачи (геометрия, материал, нормировки, модель и т.п.)

import json
from pathlib import Path
import argparse
from datetime import datetime

DEFAULT_JSON = "problem_config.json"

def main():
    ap = argparse.ArgumentParser(description="Сохранить конфигурацию задачи PINN в JSON.")
    ap.add_argument("-o", "--out", default=DEFAULT_JSON, help="Путь к JSON (по умолчанию problem_config.json)")
    args = ap.parse_args()
    config = {
        "version": "pinn_quartz_v1",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "normalization": {
            "coords": {"rho": "[0,1] (r/R)", "zeta": "[0,1] (z/H)", "tau": "[0,1] (t/T)"},
            "use_single_beta": True
        },
        "geometry_SI": {
            "W_m": 152.4e-6,     # ширина канала
            "H_m": 21.9e-6,      # глубина
            "R_m": None          # если None, при загрузке возьмем W/2
        },
        "time": {
            "Twindow_s": 50e-3   # окно времени, нормируемое в [0,1]
        },
        "material_SI": {
            "k_W_mK": 1.38,
            "rho_kg_m3": 2200.0,
            "cp_J_kgK": 703.0
        },
        "laser": {
            "P_W": 11.1,         # опционально
            "w0_m": 31.0e-6,     # лучевой радиус
            "mu_star": 100.0,    # безразмерное затухание по глубине (μ* = μ·H)
            "S_amp": 1.0         # амплитуда S' (в безразмерном уравнении)
        },
        "model": {
            "arch": {"in_dim": 3, "widths": [64, 64, 64, 64], "out_dim": 1}
        },
        "training": {
            "N_coll": 20000, "N_ic": 4096, "N_axis": 2048, "N_wall": 2048, "N_z": 2048,
            "num_epochs": 3000, "lr": 1e-3
        },
        "visualization": {
            "grid_n": 65
        },
        "paths": {
            "U_series": "U_series.npz",
            "single_field": "U_tau1.npy"
        },
        "temperature": {
            # для вывода абсолютной T: T(°C) = T0_C + DELTA_T_SCALE_K * U
            "T0_C": 25.0,
            "DELTA_T_SCALE_K": 1.0
        }
    }

    out = Path(args.out)
    out.write_text(json.dumps(config, indent=2, ensure_ascii=False, sort_keys=True))
    print(f"[OK] Конфигурация задачи сохранена в: {out.resolve()}")

if __name__ == "__main__":
    main()
