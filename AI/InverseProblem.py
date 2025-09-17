

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

def initial_gaussian(x_tensor, t0=1.0, sigma=0.1):
    center = 0.0
    return t0 * torch.exp(- (x_tensor - center)**2 / (2 * sigma**2))

class PINN(nn.Module):
    def __init__(self, layers_sizes):
        super().__init__()
        layers = []
        for i in range(len(layers_sizes) - 1):
            layers.append(nn.Linear(layers_sizes[i], layers_sizes[i + 1]))
            if i < len(layers_sizes) - 2:
                layers.append(nn.Tanh())
        self.network = nn.Sequential(*layers)

    def forward(self, x_tensor, t_tensor):
        inputs = torch.stack([x_tensor, t_tensor], dim=1)
        return self.network(inputs)

def generate_data(nx=150, nt=150, x_min=-1.0, x_max=1.0,
                  t_max=1.0, diff_coef=0.1):
    x = np.linspace(x_min, x_max, nx)
    t = np.linspace(0.0, t_max, nt)
    dx = x[1] - x[0]
    dt = t[1] - t[0]
    r = diff_coef * dt / (2 * dx**2)

    a = -r * np.ones(nx-1)
    b = (1 + 2*r) * np.ones(nx)
    c = -r * np.ones(nx-1)
    b[0], c[0] = 1 + 2*r, -2*r
    a[-1], b[-1] = -2*r, 1 + 2*r

    u = np.zeros((nx, nt))
    u[:, 0] = np.exp(- x**2 / (2 * 0.1**2))

    def thomas_solver(a, b, c, d):
        n = len(d)
        cp = np.zeros(n-1)
        dp = np.zeros(n)
        cp[0] = c[0] / b[0]
        dp[0] = d[0] / b[0]
        for i in range(1, n-1):
            denom = b[i] - a[i-1] * cp[i-1]
            cp[i] = c[i] / denom
            dp[i] = (d[i] - a[i-1] * dp[i-1]) / denom
        dp[-1] = (d[-1] - a[-2] * dp[-2]) / (b[-1] - a[-2] * cp[-2])
        x_sol = np.zeros(n)
        x_sol[-1] = dp[-1]
        for i in range(n-2, -1, -1):
            x_sol[i] = dp[i] - cp[i] * x_sol[i+1]
        return x_sol

    for n in range(nt-1):
        d = r * u[2:, n] + (1 - 2*r) * u[1:-1, n] + r * u[:-2, n]
        d = np.concatenate([
            [(1 - 2*r) * u[0, n] + 2*r * u[1, n]],
            d,
            [2*r * u[-2, n] + (1 - 2*r) * u[-1, n]]
        ])
        u[:, n+1] = thomas_solver(a, b, c, d)

    X, T = np.meshgrid(x, t, indexing='ij')
    return X.flatten(), T.flatten(), u.flatten(), x, t, u

def compute_inverse_loss(model, x_coll, t_coll, u_coll):
    D = model.coeff
    x_coll.requires_grad_(True)
    t_coll.requires_grad_(True)
    u_pred = model(x_coll, t_coll)
    u_t = torch.autograd.grad(u_pred, t_coll, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0]
    u_x = torch.autograd.grad(u_pred, x_coll, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x_coll, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    loss_pde = torch.mean((u_t - D * u_xx)**2)
    loss_data = torch.mean((u_pred - u_coll)**2)
    return loss_pde + loss_data

class PINNInverse(nn.Module):
    def __init__(self, layers_sizes, init_coef=0.1):
        super().__init__()
        self.net = PINN(layers_sizes)
        self.coeff = nn.Parameter(torch.tensor(init_coef, dtype=torch.float32))
    def forward(self, x_tensor, t_tensor):
        return self.net(x_tensor, t_tensor)

def train_inverse(model, x_train, t_train, u_train,
                  num_epochs=10000, lr=1e-3, device='cpu'):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    x_t = torch.tensor(x_train, dtype=torch.float32, device=device)
    t_t = torch.tensor(t_train, dtype=torch.float32, device=device)
    u_t = torch.tensor(u_train, dtype=torch.float32, device=device).unsqueeze(1)
    loss_history, D_history = [], []
    for epoch in range(1, num_epochs+1):
        optimizer.zero_grad()
        loss = compute_inverse_loss(model, x_t, t_t, u_t)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
        D_history.append(model.coeff.item())
        if epoch % 5 == 0:
            print(f"Epoch {epoch}/{num_epochs}, Loss: {loss:.3e}, D_est: {model.coeff.item():.5f}")
    return loss_history, D_history

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    diff_true = 0.1
    x_flat, t_flat, u_flat, x_grid, t_grid, U_true = generate_data()
    pinn_inv = PINNInverse([2, 64, 64, 64, 1], init_coef=0.02).to(device)
    print("Training inverse PINN with implicit scheme...")
    loss_hist, D_hist = train_inverse(pinn_inv, x_flat, t_flat, u_flat, device=device)
    print(f"True D: {diff_true}, Estimated D: {pinn_inv.coeff.item():.5f}")

    with torch.no_grad():
        fig = plt.figure(figsize=(6, 5))
        im = plt.imshow(U_true.T, extent=(x_grid.min(), x_grid.max(), t_grid.min(), t_grid.max()),
                        aspect='auto', origin='lower', cmap='jet')
        plt.xlabel('x')
        plt.ylabel('t')
        plt.title('Crank-Nicolson')
        plt.colorbar(im, label='u(x,t)')
        plt.tight_layout()
        plt.show()

    with torch.no_grad():
        U_pred = pinn_inv(torch.tensor(x_flat, dtype=torch.float32, device=device),
                          torch.tensor(t_flat, dtype=torch.float32, device=device)).cpu().numpy().reshape(len(x_grid), len(t_grid))
        fig = plt.figure(figsize=(6, 5))
        im = plt.imshow(U_pred.T, extent=(x_grid.min(), x_grid.max(), t_grid.min(), t_grid.max()),
                        aspect='auto', origin='lower', cmap='jet')
        plt.xlabel('x')
        plt.ylabel('t')
        plt.title('PINN Solution')
        plt.colorbar(im, label='u(x,t)')
        plt.tight_layout()
        plt.show()

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(loss_hist)
    ax1.set_yscale('log')
    ax1.set_xlabel('Epoch', fontsize=14)
    ax1.set_ylabel('Loss', color='tab:blue', fontsize=14)
    ax2 = ax1.twinx()
    ax2.plot(D_hist, linestyle='--', color='red')
    ax2.set_ylabel('Heat transfer coefficient', color='tab:red', fontsize=14)
    plt.title('')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
