# ===============================
# pinn_heat/sampling.py
# ===============================
"""
Генераторы обучающих точек для PINN в осесимметричной постановке.
Выдаёт батчи для PDE (внутренность) и BC/IC (границы/нач. условие).
"""
from __future__ import annotations

from typing import Dict, Tuple
import math
import torch

from config import Config

Tensor = torch.Tensor

# ------------------------------- #
# БАЗОВЫЕ СЭМПЛЕРЫ
# ------------------------------- #

def _to_col(x: Tensor) -> Tensor:
    """Гарантирует форму (N,1)."""
    if x.ndim == 1:
        return x.view(-1, 1)
    return x


def sample_unit(N: int, *, device: str, requires_grad: bool = True) -> Tensor:
    """Равномерно по [0,1], форма (N,1)."""
    return torch.rand(N, 1, device=device, requires_grad=requires_grad)


def sample_rho_area(N: int, *, device: str, requires_grad: bool = True, axis_bias: float = 0.0) -> Tensor:
    """Сэмплинг ρ с учётом меры в полярных координатах.
    Базово r ~ sqrt(U) даёт pdf ∝ r (равномерная плотность по площади).
    axis_bias∈[0,1): доля точек, притягиваемых к оси (больше мелких ρ).
    """
    r_area = torch.sqrt(torch.rand(N, 1, device=device))
    if axis_bias <= 0:
        r = r_area
    else:
        gamma = 0.25  # агрессивная концентрация у оси
        r_axis = torch.rand(N, 1, device=device) ** gamma
        mask = (torch.rand(N, 1, device=device) < axis_bias).float()
        r = mask * r_axis + (1 - mask) * r_area
    r.requires_grad_(requires_grad)
    return r


def sample_tau_impulse(
    N: int,
    *,
    device: str,
    t0_tau: float = 0.5,
    sigma_tau: float = 0.15,
    mix: float = 0.6,
    requires_grad: bool = True,
) -> Tensor:
    """Смесь равномерного τ и «импульсного» около t0_tau с дисперсией sigma_tau.
    mix — доля импульсной компоненты (0..1). Значения обрезаются в [0,1].
    """
    tau_uni = torch.rand(N, 1, device=device)

    normal = torch.distributions.Normal(
        torch.tensor(t0_tau, device=device), torch.tensor(sigma_tau, device=device)
    )
    tau_imp = normal.sample((N, 1)).clamp(0.0, 1.0)

    tau_mix = mix * tau_imp + (1 - mix) * tau_uni
    tau_mix.requires_grad_(requires_grad)
    return tau_mix


# ------------------------------- #
# СБОРКА БАТЧЕЙ ДЛЯ ОБУЧЕНИЯ
# ------------------------------- #

def make_training_sets(
    cfg: Config,
    *,
    device: str,
    tau_center: float = 0.5,
    tau_sigma: float = 0.15,
    tau_mix: float = 0.6,
    axis_bias: float = 0.2,
) -> Dict[str, Tuple[Tensor, ...]]:
    """Готовит батчи точек для PDE/IC/BC.

    Возвращает словарь:
        {
          "pde":  (rho_c, zeta_c, tau_c),
          "ic":   (rho_ic, zeta_ic, u_ic),            # u_ic == 0
          "axis": (zeta_axis, tau_axis),              # ρ=0
          "wall": (zeta_wall, tau_wall),              # ρ=1
          "z0":   (rho_z0,  tau_z0),                  # ζ=0
          "z1":   (rho_z1,  tau_z1),                  # ζ=1
        }
    """
    rho_c = sample_rho_area(cfg.N_coll, device=device, requires_grad=True, axis_bias=axis_bias)
    zeta_c = sample_unit(cfg.N_coll, device=device, requires_grad=True)
    tau_c = sample_tau_impulse(cfg.N_coll, device=device, t0_tau=tau_center, sigma_tau=tau_sigma, mix=tau_mix)

    rho_ic = sample_rho_area(cfg.N_ic, device=device, requires_grad=True, axis_bias=axis_bias)
    zeta_ic = sample_unit(cfg.N_ic, device=device, requires_grad=True)
    u_ic = torch.zeros(cfg.N_ic, 1, device=device)

    zeta_axis = sample_unit(cfg.N_axis, device=device, requires_grad=True)
    tau_axis = sample_tau_impulse(cfg.N_axis, device=device, t0_tau=tau_center, sigma_tau=tau_sigma, mix=tau_mix)

    zeta_wall = sample_unit(cfg.N_wall, device=device, requires_grad=True)
    tau_wall = sample_tau_impulse(cfg.N_wall, device=device, t0_tau=tau_center, sigma_tau=tau_sigma, mix=tau_mix)

    rho_z0 = sample_rho_area(cfg.N_zbc, device=device, requires_grad=True, axis_bias=axis_bias)
    tau_z0 = sample_tau_impulse(cfg.N_zbc, device=device, t0_tau=tau_center, sigma_tau=tau_sigma, mix=tau_mix)

    rho_z1 = sample_rho_area(cfg.N_zbc, device=device, requires_grad=True, axis_bias=axis_bias)
    tau_z1 = sample_tau_impulse(cfg.N_zbc, device=device, t0_tau=tau_center, sigma_tau=tau_sigma, mix=tau_mix)

    return {
        "pde": (rho_c, zeta_c, tau_c),
        "ic": (rho_ic, zeta_ic, u_ic),
        "axis": (zeta_axis, tau_axis),
        "wall": (zeta_wall, tau_wall),
        "z0": (rho_z0, tau_z0),
        "z1": (rho_z1, tau_z1),
    }


# ------------------------------- #
# ДИАГНОСТИКА СЭМПЛИНГА
# ------------------------------- #

# Вспомогательная: считает сводные статистики по тензору (min, квартили, медиана, среднее)
def _summ(name: str, x: torch.Tensor) -> str:
    x = x.detach().view(-1).cpu()
    q = torch.quantile(x, torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0]))
    return (f"{name}: N={x.numel()}  min={q[0]:.3f}  Q1={q[1]:.3f}  "
            f"med={q[2]:.3f}  Q3={q[3]:.3f}  max={q[4]:.3f}  mean={x.mean():.3f}")


def diagnose_sampling(batches: Dict[str, Tuple[Tensor, ...]], *, bins: int = 24, show_plots: bool = False) -> None:
    """Короткая диагностика распределений ρ/ζ/τ в батчах make_training_sets.
    Печатает сводку и долю точек в ключевых областях.
    Если show_plots=True — рисует простые гистограммы (matplotlib).
    """
    report = []
    if "pde" in batches:
        rho_c, zeta_c, tau_c = batches["pde"]
        report += [
            _summ("PDE.rho", rho_c),
            _summ("PDE.zeta", zeta_c),
            _summ("PDE.tau", tau_c),
        ]
        for eps in (0.01, 0.02, 0.05):
            frac_axis = (rho_c <= eps).float().mean().item()
            frac_wall = (rho_c >= (1.0 - eps)).float().mean().item()
            report.append(f"PDE.rho: frac(r<= {eps:.2f})={frac_axis:.3f}  frac(r>= {1-eps:.2f})={frac_wall:.3f}")
        for win in ((0.45, 0.55), (0.40, 0.60), (0.30, 0.70)):
            a, b = win
            frac_win = ((tau_c >= a) & (tau_c <= b)).float().mean().item()
            report.append(f"PDE.tau: frac({a:.2f}..{b:.2f})={frac_win:.3f}")

    for key in ("axis", "wall", "z0", "z1"):
        if key in batches:
            a, b = batches[key]
            report += [
                _summ(f"{key}.coord", a),
                _summ(f"{key}.tau", b),
            ]

    print("\n".join(report))

    if show_plots:
        import matplotlib.pyplot as plt
        def _hist(x, title):
            x = x.detach().view(-1).cpu().numpy()
            plt.figure()
            plt.hist(x, bins=bins)
            plt.title(title)
            plt.xlabel(title.split(".")[-1])
            plt.ylabel("count")
        if "pde" in batches:
            rho_c, zeta_c, tau_c = batches["pde"]
            _hist(rho_c, "PDE.rho"); _hist(zeta_c, "PDE.zeta"); _hist(tau_c, "PDE.tau")
        for key in ("axis", "wall", "z0", "z1"):
            if key in batches:
                a, b = batches[key]
                _hist(a, f"{key}.coord"); _hist(b, f"{key}.tau")
        plt.show()
