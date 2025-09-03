# -*- coding: utf-8 -*-
# Осесимметричная PINN для кварцевой пластины (r–z), гауссов пучок, единый β

import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# --------------------------
# 0) ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# --------------------------
def set_seed(seed=0):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def second_derivative(y, x):
    """d2y/dx2 via autograd; x.requires_grad_(True) must be set before call."""
    ones = torch.ones_like(y)
    dy_dx = torch.autograd.grad(y, x, grad_outputs=ones, create_graph=True)[0]
    d2y_dx2 = torch.autograd.grad(dy_dx, x, grad_outputs=torch.ones_like(dy_dx), create_graph=True)[0]
    return dy_dx, d2y_dx2

@torch.no_grad()
def _ensure_unit_interval(*tensors):
    for t in tensors:
        if t is None:
            continue
        assert torch.all((t >= 0.0) & (t <= 1.0)), "Expected normalized inputs in [0,1]."

def compute_inverse_loss_axisymmetric(
    model: nn.Module,
    # --- коллокации внутри области для PDE ---
    rho_coll: torch.Tensor, zeta_coll: torch.Tensor, tau_coll: torch.Tensor,
    # --- опционально: «датчики» данных u(r,z,t) ---
    u_coll: torch.Tensor | None = None,
    # --- начальное условие (IC): tau = 0 ---
    rho_ic: torch.Tensor | None = None, zeta_ic: torch.Tensor | None = None,
    u_ic: torch.Tensor | None = None,
    # --- граничные условия (BC) ---
    # ось симметрии: rho=0  (по умолчанию Неймана u_r=0)
    zeta_axis: torch.Tensor | None = None, tau_axis: torch.Tensor | None = None,
    # внешняя стенка: rho=1  (по умолчанию Неймана u_r=0)
    zeta_wall: torch.Tensor | None = None, tau_wall: torch.Tensor | None = None,
    # дно/верх по z: zeta=0 и zeta=1 (по умолчанию Неймана u_z=0)
    rho_z0: torch.Tensor | None = None, tau_z0: torch.Tensor | None = None,
    rho_z1: torch.Tensor | None = None, tau_z1: torch.Tensor | None = None,
    # --- параметры диффузии ---
    beta_single: float | None = None,   # единый безразмерный коэффициент
    beta_r: float | None = None,        # если нужно раздельно
    beta_z: float | None = None,
    # --- источник в нормированных координатах (уже с множителем T): S' = T*S ---
    source_fn = None,   # callable(rho, zeta, tau) -> tensor; если None — 0
    # --- веса лоссов ---
    w_pde: float = 1.0, w_data: float = 1.0, w_ic: float = 10.0, w_bc: float = 10.0,
    # --- численная стабилизация для 1/rho ---
    eps_axis: float = 1e-6,
):
    """
    Модель model должна принимать (rho, zeta, tau) -> u.
    Все координаты должны быть НОРМИРОВАНЫ в [0,1].
    """
    _ensure_unit_interval(rho_coll, zeta_coll, tau_coll,
                          rho_ic, zeta_ic, zeta_axis, tau_axis, zeta_wall, tau_wall,
                          rho_z0, tau_z0, rho_z1, tau_z1)

    # --- Выбор β ---
    if beta_single is not None:
        beta_r_t = torch.as_tensor(beta_single, dtype=torch.float32, device=rho_coll.device)
        beta_z_t = torch.as_tensor(beta_single, dtype=torch.float32, device=rho_coll.device)
    else:
        assert (beta_r is not None) and (beta_z is not None), "Provide (beta_r,beta_z) or beta_single."
        beta_r_t = torch.as_tensor(beta_r, dtype=torch.float32, device=rho_coll.device)
        beta_z_t = torch.as_tensor(beta_z, dtype=torch.float32, device=rho_coll.device)

    # --- PDE residual on interior collocation points ---
    rho_coll = rho_coll.detach().requires_grad_(True)
    zeta_coll = zeta_coll.detach().requires_grad_(True)
    tau_coll  = tau_coll.detach().requires_grad_(True)

    u_pred = model(rho_coll, zeta_coll, tau_coll)
    # u_tau
    du_dtau = torch.autograd.grad(u_pred, tau_coll, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0]
    # radial terms
    du_drho, d2u_drho2 = second_derivative(u_pred, rho_coll)
    inv_rho = 1.0 / torch.clamp(rho_coll, min=eps_axis)
    radial_term = d2u_drho2 + inv_rho * du_drho
    radial_term = torch.where(rho_coll < eps_axis, d2u_drho2, radial_term)  # axis fix
    # axial term
    _, d2u_dzeta2 = second_derivative(u_pred, zeta_coll)
    # source
    S_val = torch.zeros_like(u_pred) if source_fn is None else source_fn(rho_coll, zeta_coll, tau_coll)

    pde_res = du_dtau - beta_r_t * radial_term - beta_z_t * d2u_dzeta2 - S_val
    loss_pde = torch.mean(pde_res**2)

    # --- Data loss (optional) ---
    loss_data = torch.tensor(0.0, device=u_pred.device)
    if u_coll is not None:
        loss_data = torch.mean((u_pred - u_coll)**2)

    # --- Initial condition at tau=0: u(rho,zeta,0) = u_ic ---
    loss_ic = torch.tensor(0.0, device=u_pred.device)
    if (rho_ic is not None) and (zeta_ic is not None) and (u_ic is not None):
        tau0 = torch.zeros_like(rho_ic)
        u0_pred = model(rho_ic, zeta_ic, tau0)
        loss_ic = torch.mean((u0_pred - u_ic)**2)

    # --- Boundary conditions (defaults: insulated Neumann) ---
    loss_bc = torch.tensor(0.0, device=u_pred.device)

    # Axis symmetry: rho = 0 -> u_r = 0
    if (zeta_axis is not None) and (tau_axis is not None):
        rho0 = torch.zeros_like(zeta_axis).detach().requires_grad_(True)
        z0   = zeta_axis.detach().requires_grad_(True)
        t0   = tau_axis.detach().requires_grad_(True)
        u_ax = model(rho0, z0, t0)
        du_drho_ax = torch.autograd.grad(u_ax, rho0, grad_outputs=torch.ones_like(u_ax), create_graph=True)[0]
        loss_bc = loss_bc + torch.mean(du_drho_ax**2)

    # Outer wall: rho = 1 -> u_r = 0  (change if you have Dirichlet or Robin)
    if (zeta_wall is not None) and (tau_wall is not None):
        rho1 = torch.ones_like(zeta_wall).detach().requires_grad_(True)
        z1   = zeta_wall.detach().requires_grad_(True)
        t1   = tau_wall.detach().requires_grad_(True)
        u_w  = model(rho1, z1, t1)
        du_drho_w = torch.autograd.grad(u_w, rho1, grad_outputs=torch.ones_like(u_w), create_graph=True)[0]
        loss_bc = loss_bc + torch.mean(du_drho_w**2)

    # Bottom: zeta = 0 -> u_z = 0
    if (rho_z0 is not None) and (tau_z0 is not None):
        r0 = rho_z0.detach().requires_grad_(True)
        z0 = torch.zeros_like(r0).detach().requires_grad_(True)
        t0 = tau_z0.detach().requires_grad_(True)
        u_b = model(r0, z0, t0)
        du_dz_b = torch.autograd.grad(u_b, z0, grad_outputs=torch.ones_like(u_b), create_graph=True)[0]
        loss_bc = loss_bc + torch.mean(du_dz_b**2)

    # Top: zeta = 1 -> u_z = 0
    if (rho_z1 is not None) and (tau_z1 is not None):
        r1 = rho_z1.detach().requires_grad_(True)
        z1 = torch.ones_like(r1).detach().requires_grad_(True)
        t1 = tau_z1.detach().requires_grad_(True)
        u_t = model(r1, z1, t1)
        du_dz_t = torch.autograd.grad(u_t, z1, grad_outputs=torch.ones_like(u_t), create_graph=True)[0]
        loss_bc = loss_bc + torch.mean(du_dz_t**2)

    # --- Total ---
    loss_total = w_pde*loss_pde + w_data*loss_data + w_ic*loss_ic + w_bc*loss_bc
    return loss_total, {
        "loss_pde": loss_pde.detach().item(),
        "loss_data": loss_data.detach().item(),
        "loss_ic": loss_ic.detach().item(),
        "loss_bc": loss_bc.detach().item()
    }

# --------------------------
# 1) ПАРАМЕТРЫ ЗАДАЧИ
# --------------------------
set_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Геометрия из драфта (в метрах)
W = 252.4e-6          # ширина канала (по r-сечению)
R = W/2               # радиус области (ось симметрии в 0)
H = 21.9e-6           # глубина
w0 = 2.0e-6          # лучевой радиус (для пятна 62 мкм по 1/e)
Twindow = 10e-6       # окно времени, которое нормируем в [0,1]
P = 11.1              # Вт (для справки/масштаба источника, не обязателен)

# Свойства fused silica (примерные табличные)
k, rho, cp = 1.38, 2200.0, 703.0
alpha = k / (rho * cp)   # ~8.9e-7 м^2/с

# ЕДИНЫЙ безразмерный коэффициент β:
# β = α T * 0.5 * (1/R^2 + 1/H^2)
beta_single = alpha * Twindow * 0.5 * (1.0 / (R * R) + 1.0 / (H * H))

# Безразмерный лучевой радиус
w0_star = w0 / R

# Поглощение по глубине (безразмерное): mu_star = μ * H
# Если данных нет, возьмем "поверхностное" поглощение (большое μ*)
mu_star = 100.0

# Амплитуда источника в S' (безразмерная); при желании привяжите к P*

I0 = 2 * P / (math.pi * w0**2)     # пик. интенсивность, Вт/м^2
mu  = mu_star / H                  # 1/м
S_amp = (Twindow / (rho * cp)) * mu * I0
S_amp = 1


def source_fn(rho, zeta, tau):
    # Гауссов по радиусу, экспоненциальное затухание по глубине; постоянен по времени
    return S_amp * torch.exp(-2.0 * (rho / w0_star) ** 2) * torch.exp(-mu_star * zeta)

# --------------------------
# 2) МОДЕЛЬ (MLP)
# --------------------------
class MLP(nn.Module):
    def __init__(self, in_dim=3, widths=(64, 64, 64, 64), out_dim=1):
        super().__init__()
        layers = []
        d = in_dim
        for w in widths:
            layers += [nn.Linear(d, w), nn.Tanh()]
            d = w
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, rho, zeta, tau):
        x = torch.cat([rho, zeta, tau], dim=1)
        raw = self.net(x)
        return F.softplus(raw)  # гарантирует u >= 0

# --------------------------
# 3) ВЫБОРКА ТОЧЕК
# --------------------------
def sample_uniform(n):          # [0,1] -> [N,1]
    return torch.rand(n, 1, device=device)

def make_training_sets(N_coll=20000, N_ic=4096, N_axis=2048, N_wall=2048, N_z=2048):
    # Внутренние точки для PDE
    rho_c  = sample_uniform(N_coll)
    zeta_c = sample_uniform(N_coll)
    tau_c  = sample_uniform(N_coll)

    # Начальное условие: u(r,z,0)=0 (можно заменить на вашу IC)
    rho_ic  = sample_uniform(N_ic)
    zeta_ic = sample_uniform(N_ic)
    u_ic    = torch.zeros(N_ic, 1, device=device)

    # Ось симметрии (rho=0), изоляция: u_r=0
    zeta_axis = sample_uniform(N_axis)
    tau_axis  = sample_uniform(N_axis)

    # Внешняя стенка (rho=1), изоляция: u_r=0
    zeta_wall = sample_uniform(N_wall)
    tau_wall  = sample_uniform(N_wall)

    # Дно (zeta=0) и верх (zeta=1): изоляция u_z=0
    rho_z0 = sample_uniform(N_z)
    tau_z0 = sample_uniform(N_z)

    rho_z1 = sample_uniform(N_z)
    tau_z1 = sample_uniform(N_z)

    return {
        "rho_c": rho_c, "zeta_c": zeta_c, "tau_c": tau_c,
        "rho_ic": rho_ic, "zeta_ic": zeta_ic, "u_ic": u_ic,
        "zeta_axis": zeta_axis, "tau_axis": tau_axis,
        "zeta_wall": zeta_wall, "tau_wall": tau_wall,
        "rho_z0": rho_z0, "tau_z0": tau_z0,
        "rho_z1": rho_z1, "tau_z1": tau_z1
    }

# --------------------------
# 4) ОБУЧЕНИЕ
# --------------------------
def train(num_epochs=3000, print_every=200, lr=1e-3):
    data = make_training_sets()
    model = MLP().to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    for epoch in range(1, num_epochs + 1):
        optimizer.zero_grad()

        loss, parts = compute_inverse_loss_axisymmetric(
            model,
            data["rho_c"], data["zeta_c"], data["tau_c"],
            u_coll=None,                                   # если есть измерения поля, подайте сюда
            rho_ic=data["rho_ic"], zeta_ic=data["zeta_ic"], u_ic=data["u_ic"],
            zeta_axis=data["zeta_axis"], tau_axis=data["tau_axis"],
            zeta_wall=data["zeta_wall"], tau_wall=data["tau_wall"],
            rho_z0=data["rho_z0"], tau_z0=data["tau_z0"],
            rho_z1=data["rho_z1"], tau_z1=data["tau_z1"],
            beta_single=beta_single,                       # единый β
            source_fn=source_fn,
            w_pde=1.0, w_data=1.0, w_ic=10.0, w_bc=10.0,
            eps_axis=1e-6
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if epoch == 1 or epoch % print_every == 0:
            print(f"[{epoch:5d}] loss={loss.item():.4e}  "
                  f"(pde={parts['loss_pde']:.2e}, ic={parts['loss_ic']:.2e}, bc={parts['loss_bc']:.2e})")

    return model

# --------------------------
# 5) ЗАПУСК
# --------------------------
if __name__ == "__main__":
    print("Device:", device)
    if device.type == "cuda":
        torch.cuda.init()  # явная инициализация контекста (опционально)
        _ = torch.empty(1, device="cuda").sum()  # любая тривиальная CUDA-операция
        torch.cuda.synchronize()
    print(f"beta(single) = {beta_single:.3e}  (w0* = {w0_star:.3f}, mu* = {mu_star:.1f})")

    model = train(num_epochs=3000, print_every=200, lr=1e-3)
    # Проба: поле при τ=1 на сетке 65x65
    with torch.no_grad():
        n = 65
        rho = torch.linspace(0, 1, n, device=device).view(-1, 1)
        zeta = torch.linspace(0, 1, n, device=device).view(-1, 1)
        RHO, ZETA = torch.meshgrid(rho.squeeze(-1), zeta.squeeze(-1), indexing="ij")
        RHO = RHO.reshape(-1, 1); ZETA = ZETA.reshape(-1, 1)
        # --- Сохраняем серию по времени и общий архив ---
        taus = torch.tensor([0.00, 0.05, 0.1, 0.15, 0.25, 0.50, 0.75, 1.00], device=device)
        U_list = []
        for tau in taus:
            TAU = torch.full_like(RHO, tau)
            U_tau = model(RHO, ZETA, TAU).reshape(n, n).cpu().numpy()
            np.save(f"U_tau{float(tau):.2f}.npy", U_tau)  # отдельный .npy для каждого τ
            U_list.append(U_tau)

        np.savez(
            "U_series.npz",
            U=np.stack(U_list, axis=0),  # shape: [N_tau, n, n]
            tau=taus.cpu().numpy(),  # нормированные τ
            t_sec=(taus * Twindow).cpu().numpy()  # абсолютное время в секундах
        )
        # краткая сводка по последнему (τ=1):
        U_last = U_list[-1]
        print("Summary @ tau=1:",
              f"u.min={U_last.min():.3e}, u.max={U_last.max():.3e}, u.mean={U_last.mean():.3e}")
