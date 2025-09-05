
# -*- coding: utf-8 -*-
"""
measure_isotherm_width.py
Простой скрипт без argparse: считает радиус и диаметр изотермы T_iso на сечении z=z_ум.
Запускается прямо из IDE (Run/Debug). Результаты печатает и, по желанию, строит графики.
U — безразмерный; переводим в °C через параметры из pinn_params.json.
"""

from pathlib import Path
import math
import json
import numpy as np
import matplotlib.pyplot as plt

# ========= РЕДАКТИРУЕМЫЕ ПАРАМЕТРЫ =========
NPZ_PATH          = "U_series.npz"      # файл с полем U(t, r, z)
PARAMS_JSON_PATH  = "pinn_params.json"  # файл параметров задачи
T_iso_C           = None              # абсолютная температура изотермы (°C).
                                        # Если None → возьмём T0_C + 10°C из JSON.
z_target_um       = 0.0                 # глубина сечения (µm), 0 — поверхность
MAKE_PLOTS        = True                # рисовать r_iso(t) и диаметр
SHOW_PROFILES     = False               # показать несколько профилей T(r)
CSV_PATH          = "isotherm_width.csv"  # пустая строка "" — не сохранять CSV
# ==========================================


def load_params(json_path: str):
    cfg = {}
    p = Path(json_path)
    if p.exists():
        cfg = json.loads(p.read_text(encoding="utf-8"))
    else:
        print(f"[warn] {json_path} не найден. Использую значения по умолчанию.")

    # Геометрия (м)
    W_m  = cfg.get("W_m", 152.4e-6)
    H_m  = cfg.get("H_m", 21.9e-6)
    R_m  = cfg.get("R_m", W_m / 2.0)
    w0_m = cfg.get("w0_m", 2.0e-6)

    # Материал
    rho_m     = cfg.get("rho_m", 2200.0)   # кг/м^3
    cp_JkgK   = cfg.get("cp_JkgK", 703.0)  # Дж/(кг·К)

    # Время нормировки
    Twindow_s = cfg.get("Twindow_s", 50e-3)

    # Источник / оптика
    P_W       = cfg.get("P_W", 0.1)
    mu_star   = cfg.get("mu_star", 100.0)   # безразмерное (на H)
    mu_inv_m  = mu_star / H_m               # 1/м

    # Базовая температура
    T0_C      = cfg.get("T0_C", 25.0)

    return dict(W_m=W_m, H_m=H_m, R_m=R_m, w0_m=w0_m,
                rho_m=rho_m, cp_JkgK=cp_JkgK,
                Twindow_s=Twindow_s, P_W=P_W,
                mu_star=mu_star, mu_inv_m=mu_inv_m,
                T0_C=T0_C)

def delta_T_scale_K(params):
    """
    Масштаб перевода безразмерного U -> Кельвины:
    ΔT_scale = Twindow * (2 P μ) / (π w0^2 ρ cp)
    где μ = mu_star / H.
    """
    return (params["Twindow_s"] * (2.0 * params["P_W"] * params["mu_inv_m"])) / (
        math.pi * params["w0_m"]**2 * params["rho_m"] * params["cp_JkgK"]
    )

def find_first_crossing_radius(r_um, T_profile_C, T_iso_C):
    """
    Первый радиус r>=0, где T(r) пересекает T_iso_C (от центра наружу).
    Линейная интерполяция между соседними узлами. Возвращает np.nan если нет пересечения.
    """
    y = T_profile_C - T_iso_C
    for i in range(len(y) - 1):
        y0, y1 = y[i], y[i+1]
        if np.isnan(y0) or np.isnan(y1):
            continue
        if y0 == 0.0:
            return float(r_um[i])
        if y0 * y1 < 0.0 or y1 == 0.0:
            t0, t1 = T_profile_C[i], T_profile_C[i+1]
            if t1 == t0:
                return float(r_um[i])
            alpha = (T_iso_C - t0) / (t1 - t0)
            r_cross = r_um[i] + alpha * (r_um[i+1] - r_um[i])
            return float(r_cross)
    return float("nan")

def main_simple():
    # Загрузка параметров и данных
    params = load_params(PARAMS_JSON_PATH)

    # Если T_iso_C не задан, возьмём изотерму на +10°C к базовой
    iso_temp = params["T0_C"] + 10.0 if (T_iso_C is None) else float(T_iso_C)
    npz_path = Path(NPZ_PATH)
    assert npz_path.exists(), f"Файл не найден: {npz_path}"
    data = np.load(npz_path)

    assert "U" in data, "В NPZ не найден массив 'U'. Ожидается U[N_tau, n_r, n_z]."
    U = data["U"]  # (N_tau, n_r, n_z)

    # Время
    if "t_sec" in data:
        t_sec = data["t_sec"]
        tau   = data["tau"] if "tau" in data else None
    else:
        assert "tau" in data, "Нет ни 't_sec', ни 'tau' в NPZ."
        tau   = data["tau"]
        t_sec = tau * float(params["Twindow_s"])

    N_tau, n_r, n_z = U.shape

    # Сетка (в µm)
    R_um = params["R_m"] * 1e6
    H_um = params["H_m"] * 1e6
    r_um = np.linspace(0.0, R_um, n_r)
    z_um = np.linspace(0.0, H_um, n_z)

    # Выбор слоя по z
    iz = int(np.clip(np.argmin(np.abs(z_um - z_target_um)), 0, n_z - 1))
    if abs(z_um[iz] - z_target_um) > 1e-9:
        print(f"[info] Использую ближайший слой по z: {z_um[iz]:.3f} µm (запрошено {z_target_um:.3f} µm)")

    # Перевод U -> T (°C): ТОЛЬКО через ΔT_scale (U безразмерный!)
    dT_K = float(delta_T_scale_K(params))
    T_C = params["T0_C"] + dT_K * U

    # Поиск r_iso и диаметра
    r_iso_um = np.full(N_tau, np.nan, dtype=float)
    diam_um  = np.full(N_tau, np.nan, dtype=float)

    for k in range(N_tau):
        profile = T_C[k, :, iz]
        r_cross = find_first_crossing_radius(r_um, profile, iso_temp)
        r_iso_um[k] = r_cross
        if np.isfinite(r_cross):
            diam_um[k] = 2.0 * r_cross

    # Итоговая сводка
    n_found = int(np.isfinite(r_iso_um).sum())
    print(f"[ok] Изотерма T={iso_temp:.3f} °C на z={z_um[iz]:.3f} µm найдена в {n_found} из {N_tau} моментов времени.")
    print("Примеры (первые 5):")
    for k in range(min(5, N_tau)):
        r_str = f"{r_iso_um[k]:.3f} µm" if np.isfinite(r_iso_um[k]) else "—"
        d_str = f"{diam_um[k]:.3f} µm"  if np.isfinite(diam_um[k])  else "—"
        t_str = f"{t_sec[k]*1e6:.3f} µs"
        print(f"  k={k:3d}  t={t_str:>10}  r_iso={r_str:>10}  diam={d_str:>10}")

    # CSV (по желанию)
    if CSV_PATH:
        import csv
        with Path(CSV_PATH).open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, delimiter=";")
            w.writerow(["k", "t_sec", "tau", "z_um", "T_iso_C", "r_iso_um", "diameter_um"])
            for k in range(N_tau):
                w.writerow([
                    k,
                    float(t_sec[k]),
                    (float(tau[k]) if tau is not None else ""),
                    float(z_um[iz]),
                    float(iso_temp),
                    (float(r_iso_um[k]) if np.isfinite(r_iso_um[k]) else ""),
                    (float(diam_um[k])  if np.isfinite(diam_um[k])  else "")
                ])
        print(f"[ok] CSV сохранён: {CSV_PATH}")

    # Графики (по желанию)
    if MAKE_PLOTS:
        # переводим в микросекунды
        t_us = t_sec * 1e6

        plt.figure(figsize=(6, 4))
        plt.plot(t_us, r_iso_um, marker="o", lw=1)
        plt.xlabel("t, µs")
        plt.ylabel("r_iso, µm")
        plt.title(f"Радиус изотермы T={iso_temp:.3f} °C при z={z_um[iz]:.3f} µm")
        plt.grid(True)
        plt.tight_layout()

        plt.figure(figsize=(6, 4))
        plt.plot(t_us, diam_um, marker="o", lw=1)
        plt.xlabel("t, µs")
        plt.ylabel("Диаметр изотермы, µm")
        plt.title(f"Ширина (диаметр) изотермы T={iso_temp:.3f} °C при z={z_um[iz]:.3f} µm")
        plt.grid(True)
        plt.tight_layout()

        if SHOW_PROFILES:
            sel = np.linspace(0, N_tau - 1, min(4, N_tau)).round().astype(int)
            for k in sel:
                plt.figure(figsize=(6, 4))
                plt.plot(r_um, T_C[k, :, iz], lw=1.5, label=f"k={k}, t={t_sec[k]*1e6:.3f} µs")
                plt.axhline(iso_temp, ls="--", label=f"T_iso={iso_temp:.3f} °C")
                if np.isfinite(r_iso_um[k]):
                    plt.axvline(r_iso_um[k], ls=":", label=f"r_iso={r_iso_um[k]:.2f} µm")
                plt.xlabel("r, µm")
                plt.ylabel("T, °C")
                plt.title(f"Профиль T(r) при z={z_um[iz]:.3f} µm")
                plt.legend()
                plt.grid(True)
                plt.tight_layout()

        plt.show()

if __name__ == "__main__":
    main_simple()



