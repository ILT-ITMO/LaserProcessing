# ===============================
# pinn_heat/physics.py
# ===============================
"""
Физика и лоссы для осесимметричного теплопереноса в безразмерных переменных.

Уравнение:
    U_τ = β_r (U_ρρ + (1/ρ) U_ρ) + β_z U_ζζ + S(ρ, ζ, τ)

Где:
  β_r = a * Twindow / R^2, β_z = a * Twindow / H^2,
  a = k / (ρ_material * c_p) — термодиффузивность.

Источник S(ρ,ζ,τ) подаётся извне и ДОЛЖЕН быть безразмерным.
Эту нормировку выполняет source.py, используя DELTA_T_SCALE_K из пресета.
"""

from __future__ import annotations
from typing import Callable, Dict, Tuple, Optional, Union
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# ---- Плоский импорт без пакета ----
from config import Config  # вместо from .config import Config

Tensor = torch.Tensor
SourceLike = Optional[Union[nn.Module, Callable[..., Tensor]]]


def physical_estimation(cfg:Config):
    rho_kg_m3 = float(cfg.rho_kg_m3)
    cp  = float(cfg.cp_J_kgK)
    k = float(cfg.k_W_mK)
    Tw  = float(cfg.Twindow_s)
    w0  = float(cfg.w0_m)
    H   = float(cfg.H_m)
    eta = float(getattr(cfg, "eta_abs", 1.0))
    wl = float(cfg.wl)
    k_imag = float(cfg.k_imag)
    # Средняя мощность для расчёта (если нужна)
    P_W = float(cfg.P_W)
    alpha = k/rho_kg_m3/cp
    mu = 4*np.pi*k_imag/wl
    rep_rate = cfg.rep_rate_Hz
    t_char_heat = H ** 2 / alpha  # Характерное время прогрева
    pulse_duration = cfg.pulse_duration_s
    l_diff = np.sqrt(alpha*pulse_duration)

    print(f"Характерное время прогрева слоя {t_char_heat*1e6:.2f} us толщиной {H*1e6} um")
    print(f"Период между импульсами {1/rep_rate*1e6} us")
    print(f"Характерная диффузионная длина {l_diff}")
    if 1/rep_rate < t_char_heat:
        print('Происходит локальное накопление энергии')
    else:
        print('Тепло успевает диссипировать')
    if l_diff <
        print('Пользоваться cильным поглощением нельзя')
    # print(f"")
    # print(mu)
    t_char_heat = H**2/alpha # Характерное время прогрева
    return 0

# ------------------------------- #
# Коэффициенты безразмерности
# ------------------------------- #

def thermal_diffusivity(cfg: Config) -> float:
    """a = k / (rho * cp)  [м^2/с]."""
    return float(cfg.k_W_mK / (cfg.rho_kg_m3 * cfg.cp_J_kgK))


def nondim_coeffs(cfg: Config) -> Tuple[float, float]:
    """
    β_r, β_z для ρ=r/R, ζ=z/H, τ=t/Twindow.
    """
    a = thermal_diffusivity(cfg)
    beta_r = a * cfg.Twindow_s / (cfg.R_m ** 2)
    beta_z = a * cfg.Twindow_s / (cfg.H_m ** 2)
    return float(beta_r), float(beta_z)


# ------------------------------- #
# Автодифференцирование
# ------------------------------- #

def _grad(u: Tensor, x: Tensor) -> Tensor:
    (g,) = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)
    return g


def _grad2(u: Tensor, x: Tensor) -> Tensor:
    ux = _grad(u, x)
    (g2,) = torch.autograd.grad(ux, x, grad_outputs=torch.ones_like(ux), create_graph=True)
    return g2


# ------------------------------- #
# Вызов источника (совместимость с source.py)
# ------------------------------- #

def _eval_source(
    src: SourceLike,
    rho: Tensor,
    zeta: Tensor,
    tau: Tensor,
    cfg: Config,
) -> Tensor:
    """
    Унифицированный вызов источника:
      - nn.Module с forward(rho, zeta, tau)  -> как в OpticalSource
      - callable(rho, zeta, tau)             -> без cfg
      - callable(rho, zeta, tau, cfg)        -> с cfg
    Если src=None — возвращает нули.
    """
    if src is None:
        return torch.zeros_like(rho)

    if isinstance(src, nn.Module):
        return src(rho, zeta, tau)

    try:
        return src(rho, zeta, tau)  # type: ignore[misc]
    except TypeError:
        return src(rho, zeta, tau, cfg)  # type: ignore[misc]


def make_source_from_json(params_json_path: str | Path, *, device: Optional[str] = None) -> nn.Module:
    """
    Построить совместимый источник S(ρ,ζ,τ) из JSON пресета.
    Оборачивает source.build_optical_source, импортируя его только при вызове.
    """
    from source import build_optical_source  # локальный импорт, чтобы не плодить циклы
    return build_optical_source(Path(params_json_path), device=device)


# ------------------------------- #
# PDE-резидуал
# ------------------------------- #

def pde_residual(
    model: nn.Module,
    rho: Tensor,
    zeta: Tensor,
    tau: Tensor,
    cfg: Config,
    *,
    source: SourceLike = None,
    eps_axis: float = 1e-6,
) -> Tensor:
    """
    r = U_τ - β_r (U_ρρ + (1/ρ) U_ρ) - β_z U_ζζ - S(ρ, ζ, τ)

    Примечание: (1/ρ)U_ρ стабилизируем через clamp(ρ, eps_axis)
    + отдельный осевой BC.
    """
    beta_r, beta_z = nondim_coeffs(cfg)

    U = model(rho, zeta, tau)

    U_tau = _grad(U, tau)
    U_rho = _grad(U, rho)
    U_zeta = _grad(U, zeta)

    U_rr = _grad2(U, rho)
    U_zz = _grad2(U, zeta)

    rho_safe = torch.clamp(rho, min=eps_axis)
    lap_ax = U_rr + (U_rho / rho_safe)

    S_hat = _eval_source(source, rho, zeta, tau, cfg)

    resid = U_tau - (beta_r * lap_ax + beta_z * U_zz) - S_hat
    return resid


# ------------------------------- #
# Резидуалы IC / BC
# ------------------------------- #

def ic_residual(
    model: nn.Module,
    rho_ic: Tensor,
    zeta_ic: Tensor,
    tau_ic: Tensor,       # обычно нули
    u_ic_target: Tensor,  # обычно нули
) -> Tensor:
    U0 = model(rho_ic, zeta_ic, tau_ic)
    return U0 - u_ic_target


def axis_neumann_residual(model: nn.Module, zeta: Tensor, tau: Tensor) -> Tensor:
    """∂U/∂ρ |_{ρ=0} = 0."""
    rho0 = torch.zeros_like(zeta, requires_grad=True, device=zeta.device)
    U = model(rho0, zeta, tau)
    U_rho = _grad(U, rho0)
    return U_rho


def wall_neumann_residual(model: nn.Module, zeta: Tensor, tau: Tensor) -> Tensor:
    """∂U/∂ρ |_{ρ=1} = 0."""
    rho1 = torch.ones_like(zeta, requires_grad=True, device=zeta.device)
    U = model(rho1, zeta, tau)
    U_rho = _grad(U, rho1)
    return U_rho


def z0_neumann_residual(model: nn.Module, rho: Tensor, tau: Tensor) -> Tensor:
    """∂U/∂ζ |_{ζ=0} = 0."""
    z0 = torch.zeros_like(rho, requires_grad=True, device=rho.device)
    U = model(rho, z0, tau)
    U_z = _grad(U, z0)
    return U_z


def z1_neumann_residual(model: nn.Module, rho: Tensor, tau: Tensor) -> Tensor:
    """∂U/∂ζ |_{ζ=1} = 0."""
    z1 = torch.ones_like(rho, requires_grad=True, device=rho.device)
    U = model(rho, z1, tau)
    U_z = _grad(U, z1)
    return U_z


# ------------------------------- #
# Агрегация лоссов
# ------------------------------- #

def compute_losses(
    model: nn.Module,
    batches: Dict[str, Tuple[Tensor, ...]],
    cfg: Config,
    *,
    source: SourceLike = None,
) -> Dict[str, Tensor]:
    """
    Возвращает словарь лоссов:
      loss_pde, loss_ic, loss_bc_axis, loss_bc_wall, loss_bc_z0, loss_bc_z1, loss_bc, loss_total
    Весовые коэффициенты берутся из cfg: w_pde, w_ic, w_bc.
    """
    # PDE
    rho_c, zeta_c, tau_c = batches["pde"]
    if not rho_c.requires_grad:  rho_c = rho_c.detach().requires_grad_(True)
    if not zeta_c.requires_grad: zeta_c = zeta_c.detach().requires_grad_(True)
    if not tau_c.requires_grad:  tau_c = tau_c.detach().requires_grad_(True)
    r_pde = pde_residual(model, rho_c, zeta_c, tau_c, cfg, source=source)
    loss_pde = torch.mean(r_pde ** 2)

    # IC
    rho_ic, zeta_ic, u_ic_target = batches["ic"]
    tau0 = torch.zeros_like(u_ic_target, device=u_ic_target.device, requires_grad=True)
    r_ic = ic_residual(model, rho_ic, zeta_ic, tau0, u_ic_target)
    loss_ic = torch.mean(r_ic ** 2)

    # BC: ось / стенка
    zeta_axis, tau_axis = batches["axis"]
    r_axis = axis_neumann_residual(model, zeta_axis, tau_axis)
    loss_bc_axis = torch.mean(r_axis ** 2)

    zeta_wall, tau_wall = batches["wall"]
    r_wall = wall_neumann_residual(model, zeta_wall, tau_wall)
    loss_bc_wall = torch.mean(r_wall ** 2)

    # BC: ζ=0, ζ=1
    rho_z0, tau_z0 = batches["z0"]
    r_z0 = z0_neumann_residual(model, rho_z0, tau_z0)
    loss_bc_z0 = torch.mean(r_z0 ** 2)

    rho_z1, tau_z1 = batches["z1"]
    r_z1 = z1_neumann_residual(model, rho_z1, tau_z1)
    loss_bc_z1 = torch.mean(r_z1 ** 2)

    loss_bc = (loss_bc_axis + loss_bc_wall + loss_bc_z0 + loss_bc_z1) / 4.0
    loss_total = cfg.w_pde * loss_pde + cfg.w_ic * loss_ic + cfg.w_bc * loss_bc

    return dict(
        loss_pde=loss_pde,
        loss_ic=loss_ic,
        loss_bc_axis=loss_bc_axis,
        loss_bc_wall=loss_bc_wall,
        loss_bc_z0=loss_bc_z0,
        loss_bc_z1=loss_bc_z1,
        loss_bc=loss_bc,
        loss_total=loss_total,
    )

# if __name__ == "main":

BASE_CFG = dict(
    # Геометрия/нормировки
    R_m=300e-6,
    H_m=100e-6,
    w0_m=62e-6,
    Twindow_s=1e-6,
    mu_star=1.0,
    T0_C=20.0,

    # Материал (кварц)
    rho_kg_m3=2200.0,
    cp_J_kgK=740.0,
    k_W_mK=1.4,

    # Нормировка температуры
    temp_scaling_mode="A",  # "A" | "B" | "C"
    eta_abs=1.0,
    U_target=0.9,
    kappa_r=0.5, kappa_z=0.5,

    # Импульсные поля (опционально; можно оставить None)
    pulse_duration_s=15e-6,  # FWHM, s
    rep_rate_Hz=8e3,  # Hz
    pulse_count=3,  # N импульсов
    pulses_t0_s=1e-6,  # стартовая задержка
    E_pulse_J=None,  # J
    # P_W зададим перебором ниже

    # Сэмплинг/обучение
    N_coll=20000, N_ic=4096, N_axis=2048, N_wall=2048, N_zbc=1024,
    widths=(64, 64, 64, 64), lr=1e-3,
    w_pde=1.0, w_data=1.0, w_ic=10.0, w_bc=10.0,
    seed=42, device="cpu",
)

cfg = Config(**BASE_CFG)
physical_estimation(cfg)