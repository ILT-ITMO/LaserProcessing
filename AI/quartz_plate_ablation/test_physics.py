import torch
import torch.nn as nn
from types import SimpleNamespace

# === Импорты тестируемого модуля ===
from physics import (
    nondim_coeffs,
    pde_residual,
    axis_neumann_residual,
    wall_neumann_residual,
    z0_neumann_residual,
    z1_neumann_residual,
    ic_residual,
)

# ---- Вспом. "Config" (если у тебя уже есть Config из config.py — подставь его) ----
class DummyCfg(SimpleNamespace):
    pass

def make_cfg():
    # Типичные параметры кварца (примерные; важен лишь знак/масштаб для тестов)
    return DummyCfg(
        k_W_mK=1.38,          # Вт/(м·К)
        rho_kg_m3=2200.0,     # кг/м^3
        cp_J_kgK=730.0,       # Дж/(кг·К)
        R_m=100e-6,           # радиус нормировки, м
        H_m=200e-6,           # глубина нормировки, м
        Twindow_s=200e-6,     # окно по времени, с
        w_pde=1.0, w_ic=1.0, w_bc=1.0
    )

# ---- Модель, реализующая "произведённое" аналитическое U(ρ,ζ,τ) ----
class ManufacturedU(nn.Module):
    """
    U(ρ,ζ,τ) = τ * [ ρ^2 (1-ρ^2)^2 ] * [ ζ^2 (1-ζ)^2 ]
    - осесимметрично,
    - ∂U/∂ρ |_{ρ=0,1} = 0,
    - ∂U/∂ζ |_{ζ=0,1} = 0,
    - при τ=0 -> U=0 (удобно для IC).
    """
    def forward(self, rho, zeta, tau):
        rho2 = rho**2
        zeta2 = zeta**2
        fr = rho2 * (1.0 - rho2)**2
        fz = zeta2 * (1.0 - zeta)**2
        return tau * fr * fz

# ---- Источник, «согласованный» с выбранным U (MMS): S = U_τ - β_r( U_ρρ + U_ρ/ρ ) - β_z U_ζζ ----
def make_mms_source(model, cfg):
    beta_r, beta_z = nondim_coeffs(cfg)

    def S(rho, zeta, tau):
        U = model(rho, zeta, tau)

        # Первые производные
        U_tau  = torch.autograd.grad(U, tau,  torch.ones_like(U), create_graph=True)[0]
        U_rho  = torch.autograd.grad(U, rho,  torch.ones_like(U), create_graph=True)[0]
        U_zeta = torch.autograd.grad(U, zeta, torch.ones_like(U), create_graph=True)[0]

        # Вторые производные
        U_rr = torch.autograd.grad(U_rho, rho,  torch.ones_like(U_rho), create_graph=True)[0]
        U_zz = torch.autograd.grad(U_zeta, zeta, torch.ones_like(U_zeta), create_graph=True)[0]

        rho_safe = torch.clamp(rho, min=1e-6)
        lap_ax = U_rr + U_rho / rho_safe

        # Конструируем S так, чтобы PDE резидуал был нулём
        return U_tau - (beta_r * lap_ax + beta_z * U_zz)

    return S

# ---- Сэмплирование тестовых точек ----
def sample_points(N, device):
    # ρ∈(0,1], ζ∈[0,1], τ∈[0,1]
    rho  = torch.rand(N, 1, device=device, requires_grad=True) * 0.999 + 0.001
    zeta = torch.rand(N, 1, device=device, requires_grad=True)
    tau  = torch.rand(N, 1, device=device, requires_grad=True)
    return rho, zeta, tau

def run_all(device="cpu"):
    torch.manual_seed(0)
    cfg = make_cfg()

    # 1) Коэффициенты безразмерности
    beta_r, beta_z = nondim_coeffs(cfg)
    assert beta_r > 0 and beta_z > 0, "β_r, β_z должны быть положительными"

    # 2) PDE (MMS): резидуал должен быть близок к нулю
    model = ManufacturedU().to(device)
    N = 4096
    rho, zeta, tau = sample_points(N, device)
    source = make_mms_source(model, cfg)

    r = pde_residual(model, rho, zeta, tau, cfg, source=source)
    pde_mse = torch.mean(r**2).item()
    print(f"PDE MSE (MMS): {pde_mse:.3e}")
    assert pde_mse < 1e-10, "PDE резидуал слишком велик — проверь вычисление операторов"

    # 3) BC Neumann (ось, стенка, z=0, z=1)
    #    Для выбранного U все эти производные обращаются в ноль на границах.
    z_rand = torch.rand(N, 1, device=device, requires_grad=True)
    t_rand = torch.rand(N, 1, device=device, requires_grad=True)
    r_axis = axis_neumann_residual(model, z_rand, t_rand)
    r_wall = wall_neumann_residual(model, z_rand, t_rand)
    bc_axis_mse = torch.mean(r_axis**2).item()
    bc_wall_mse = torch.mean(r_wall**2).item()
    print(f"BC MSE axis: {bc_axis_mse:.3e}, wall: {bc_wall_mse:.3e}")
    assert bc_axis_mse < 1e-12 and bc_wall_mse < 1e-12, "Neumann по ρ не нулевой"

    r_z0 = z0_neumann_residual(model, torch.rand(N,1,device=device,requires_grad=True), t_rand)
    r_z1 = z1_neumann_residual(model, torch.rand(N,1,device=device,requires_grad=True), t_rand)
    bc_z0_mse = torch.mean(r_z0**2).item()
    bc_z1_mse = torch.mean(r_z1**2).item()
    print(f"BC MSE z0: {bc_z0_mse:.3e}, z1: {bc_z1_mse:.3e}")
    assert bc_z0_mse < 1e-12 and bc_z1_mse < 1e-12, "Neumann по ζ не нулевой"

    # 4) IC: при τ=0 целевое U=0 ⇒ IC резидуал ≈0
    rho_ic  = torch.rand(N, 1, device=device, requires_grad=True)
    zeta_ic = torch.rand(N, 1, device=device, requires_grad=True)
    tau0    = torch.zeros(N, 1, device=device, requires_grad=True)
    u_ic_target = torch.zeros(N, 1, device=device)

    r_ic = ic_residual(model, rho_ic, zeta_ic, tau0, u_ic_target)
    ic_mse = torch.mean(r_ic**2).item()
    print(f"IC MSE: {ic_mse:.3e}")
    assert ic_mse < 1e-20, "IC резидуал не нулевой при τ=0"

    print("OK: physics.py прошёл базовые проверки на", device)

if __name__ == "__main__":
    run_all("cpu")
    if torch.cuda.is_available():
        run_all("cuda")
