# -*- coding: utf-8 -*-
# Визуализация серии полей U(r,z,τ) из U_series.npz в абсолютных °C
# Оси: r (μm) × z (μm). Каждая панель — свой момент времени.

import math
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

NPZ_PATH = "U_series.npz"

# === НОВОЕ: грузим общие параметры из JSON, если есть ===
import json, os
from pathlib import Path

PARAMS_JSON = "pinn_params.json"
cfg = {}
if Path(PARAMS_JSON).exists():
    cfg = json.loads(Path(PARAMS_JSON).read_text())
else:
    print(f"[warn] {PARAMS_JSON} не найден. Использую значения по умолчанию.")

# Геометрия (м)
W_m  = cfg.get("W_m", 152.4e-6)
H_m  = cfg.get("H_m", 21.9e-6)
R_m  = cfg.get("R_m", W_m/2.0)
w0_m = cfg.get("w0_m", 2.0e-6)

# Материал
rho_m     = cfg.get("rho_m", 2200.0)    # кг/м^3
cp_JkgK   = cfg.get("cp_JkgK", 703.0)   # Дж/(кг·К)

# Время нормировки
Twindow_s = cfg.get("Twindow_s", 50e-3) # сек

# Оптика / мощность
P_W     = cfg.get("P_W", 0.1)           # Вт
mu_star = cfg.get("mu_star", 100.0)     # безразм. (на H)
mu_inv_m = mu_star / H_m                # 1/м

# Базовая температура и изотерма
T0_C        = cfg.get("T0_C", 25.0)
ISO_TEMP_C  = cfg.get("ISO_TEMP_C", T0_C + 10.0)

# Перевод безразмерной U -> Кельвины: ΔT_scale_K(P)
import math
DELTA_T_SCALE_K = (Twindow_s * (2.0 * P_W * mu_inv_m)) / (math.pi * w0_m**2 * rho_m * cp_JkgK)

# Если в NPZ нет 't_sec', запасной Twindow:
Twindow_fallback_s = Twindow_s

# Ограничение числа панелей (опционально)
MAX_PANELS = 8


def main():
    assert Path(NPZ_PATH).exists(), f"Файл не найден: {NPZ_PATH}"
    data = np.load(NPZ_PATH)

    assert "U" in data, "В NPZ не найден массив 'U'."
    U = data["U"]  # shape: (N_tau, n_r, n_z)

    # Время
    if "t_sec" in data:
        t_sec = data["t_sec"]
        tau = data["tau"] if "tau" in data else None
    else:
        assert "tau" in data, "Нет ни 't_sec', ни 'tau'."
        tau = data["tau"]
        t_sec = tau * float(Twindow_fallback_s)

    N_tau, n_r, n_z = U.shape

    # Перевод в °C
    T_C = T0_C + float(DELTA_T_SCALE_K) * U

    vmin = np.min(T_C)
    vmax = np.max(T_C)

    if (MAX_PANELS is not None) and (N_tau > MAX_PANELS):
        idx = np.linspace(0, N_tau - 1, MAX_PANELS).round().astype(int)
    else:
        idx = np.arange(N_tau, dtype=int)

    K = len(idx)
    ncols = min(4, K)
    nrows = math.ceil(K / ncols)

    R_um = R_m * 1e6
    H_um = H_m * 1e6
    extent = [0.0, R_um, 0.0, H_um]

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5*ncols, 3.6*nrows), constrained_layout=True)
    if K == 1:
        axes = np.array([axes])
    axes = axes.ravel()

    ims = []
    for k, ax in zip(idx, axes):
        im = ax.imshow(
            T_C[k].T, extent=extent, origin="lower", aspect="auto",
            vmin=vmin, vmax=vmax, interpolation="nearest"
        )
        ims.append(im)

        # --- Добавляем красный контур изотермы ---
        r = np.linspace(0, R_um, n_r)
        z = np.linspace(0, H_um, n_z)
        R_grid, Z_grid = np.meshgrid(r, z, indexing="ij")
        cs = ax.contour(R_grid, Z_grid, T_C[k], levels=[ISO_TEMP_C], colors="red", linewidths=1.5)
        if cs.allsegs[0]:  # если контур существует
            ax.clabel(cs, fmt=f"{ISO_TEMP_C:.1f} °C", inline=True, fontsize=8, colors="red")

        # Заголовок
        title = []
        if "tau" in data:
            title.append(f"τ = {data['tau'][k]:.2f}")
        title.append(f"t = {t_sec[k]*1e3:.1f} ms")
        ax.set_title(", ".join(title))

        ax.set_xlabel("r, μm")
        ax.set_ylabel("z, μm")

    for ax in axes[K:]:
        ax.axis("off")

    cbar = fig.colorbar(ims[0], ax=axes[:K], shrink=0.95)
    cbar.set_label("T, °C")

    plt.show()


if __name__ == "__main__":
    main()
