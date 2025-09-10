# visualize_sampling_points.py
"""
Визуализация сэмплированных точек PINN по типам условий.
Рисует scatter в координатах (rho, zeta), разными цветами для:
- PDE (внутренность домена)
- IC (начальные условия)
- axis (rho=0), wall (rho=1), z0 (zeta=0), z1 (zeta=1)
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from pinn_io import load_params_json
from sampling import make_training_sets

# ==== НАСТРОЙКИ ====
PARAMS_PATH = Path("./presets_params/pinn_params_P3p3W_V40mms_20250909_170950.json")  # путь к JSON с параметрами
DEVICE = None                           # можно задать "cpu" или "cuda", если None → cfg.device
MAX_PER_SET = 4000                      # макс. число точек каждого типа для отрисовки (-1 = без ограничения)
SEED = 123                              # seed для случайного подвыборочного отображения
SHOW_LEGEND = True                      # показывать легенду на графике
# ====================


def _to_np(x: torch.Tensor) -> np.ndarray:
    return x.detach().view(-1).cpu().numpy()


def _subsample(x: np.ndarray, y: np.ndarray, max_n: int, rng: np.random.Generator):
    n = x.shape[0]
    if max_n is None or n <= max_n:
        return x, y
    idx = rng.choice(n, size=max_n, replace=False)
    return x[idx], y[idx]


def main():
    cfg = load_params_json(PARAMS_PATH)
    device = DEVICE or cfg.device
    batches = make_training_sets(cfg, device=device)
    rng = np.random.default_rng(SEED)
    max_per = None if MAX_PER_SET is not None and MAX_PER_SET < 0 else MAX_PER_SET
    sets = {}
    rho_c, zeta_c, _ = batches["pde"]
    sets["PDE (interior)"] = (_to_np(rho_c), _to_np(zeta_c))

    rho_ic, zeta_ic, _u_ic = batches["ic"]
    sets["IC (tau=0)"] = (_to_np(rho_ic), _to_np(zeta_ic))

    zeta_axis, _tau_axis = batches["axis"]
    sets["BC axis (rho=0)"] = (np.zeros_like(_to_np(zeta_axis)), _to_np(zeta_axis))

    zeta_wall, _tau_wall = batches["wall"]
    sets["BC wall (rho=1)"] = (np.ones_like(_to_np(zeta_wall)), _to_np(zeta_wall))

    rho_z0, _tau_z0 = batches["z0"]
    sets["BC z0 (zeta=0)"] = (_to_np(rho_z0), np.zeros_like(_to_np(rho_z0)))

    rho_z1, _tau_z1 = batches["z1"]
    sets["BC z1 (zeta=1)"] = (_to_np(rho_z1), np.ones_like(_to_np(rho_z1)))

    style = {
        "PDE (interior)":   dict(color="#9aa0a6", marker=".",   s=6,  alpha=0.6),
        "IC (tau=0)":       dict(color="#1f77b4", marker="o",   s=10, alpha=0.7, edgecolors="none"),
        "BC axis (rho=0)":  dict(color="#d62728", marker="x",   s=18, alpha=0.9),
        "BC wall (rho=1)":  dict(color="#2ca02c", marker="+",   s=18, alpha=0.9),
        "BC z0 (zeta=0)":   dict(color="#ff7f0e", marker="^",   s=14, alpha=0.8),
        "BC z1 (zeta=1)":   dict(color="#9467bd", marker="v",   s=14, alpha=0.8),
    }

    plt.figure(figsize=(7.2, 7.2), dpi=110)
    for name, (rx, zx) in sets.items():
        rx, zx = _subsample(rx, zx, max_per, rng)
        kw = style.get(name, dict(marker=".", s=8, alpha=0.7))
        plt.scatter(rx, zx, label=name, **kw)

    plt.title("PINN sampling points by condition type")
    plt.xlabel(r"$\rho$ (radial, normalized 0..1)")
    plt.ylabel(r"$\zeta$ (depth, normalized 0..1)")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.grid(True, alpha=0.25)
    if SHOW_LEGEND:
        plt.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.show()

# мультипанельная визуализация: 6 отдельных графиков
import torch
import matplotlib.pyplot as plt
from sampling import make_training_sets

def visualize_sampling_panels(cfg, device="cpu", max_per_set=4000, seed=123):
    """
    Рисует 6 панелей (PDE, IC, axis, wall, z0, z1), каждая — scatter (rho, zeta).
    """
    batches = make_training_sets(cfg, device=device)
    rng = np.random.default_rng(seed)
    max_per = None if (max_per_set is not None and max_per_set < 0) else max_per_set

    # Подготовка наборов
    sets = []
    rho_c, zeta_c, _ = batches["pde"];     sets.append(("PDE (interior)", _to_np(rho_c), _to_np(zeta_c)))
    rho_ic, zeta_ic, _ = batches["ic"];    sets.append(("IC (tau=0)",    _to_np(rho_ic), _to_np(zeta_ic)))
    zeta_axis, _ = batches["axis"];        sets.append(("BC axis ρ=0",   np.zeros_like(_to_np(zeta_axis)), _to_np(zeta_axis)))
    zeta_wall, _ = batches["wall"];        sets.append(("BC wall ρ=1",   np.ones_like(_to_np(zeta_wall)),  _to_np(zeta_wall)))
    rho_z0, _ = batches["z0"];             sets.append(("BC z0 ζ=0",     _to_np(rho_z0), np.zeros_like(_to_np(rho_z0))))
    rho_z1, _ = batches["z1"];             sets.append(("BC z1 ζ=1",     _to_np(rho_z1), np.ones_like(_to_np(rho_z1))))

    fig, axes = plt.subplots(2, 3, figsize=(10, 6.5), dpi=110, constrained_layout=True)
    axes = axes.ravel()

    style = dict(marker=".", s=8, alpha=0.7)
    for ax, (name, rx, zx) in zip(axes, sets):
        rx, zx = _subsample(rx, zx, max_per, rng)
        ax.scatter(rx, zx, **style)
        ax.set_title(name, fontsize=10)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)
        ax.set_xlabel(r"$\rho$")
        ax.set_ylabel(r"$\zeta$")

    fig.suptitle("PINN sampling points by condition type (panels)", fontsize=12)
    plt.show()


if __name__ == "__main__":
    cfg = load_params_json(PARAMS_PATH)
    visualize_sampling_panels(cfg)
