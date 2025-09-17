import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

NUM_GAUSSIANS = 2       # количество гауссов
GAUSSIAN_SPACING = 0.5  # расстояние между центрами гауссов по оси x
SIGMA0 = 0.1            # ширина гауссов

def compute_centers(n, spacing, device):
    if n % 2 == 1:
        idx = torch.arange(-(n//2), n//2 + 1, device=device, dtype=torch.float32)
    else:
        half = n // 2
        idx = torch.arange(-half + 0.5, half, step=1.0, device=device, dtype=torch.float32)
    return idx * spacing

def initial_gaussian(x_tensor, t0=1.0):
    centers = compute_centers(NUM_GAUSSIANS, GAUSSIAN_SPACING, x_tensor.device)
    gaussians = [t0 * torch.exp(- (x_tensor - c)**2 / (2 * SIGMA0**2)) for c in centers]
    return sum(gaussians)

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

def compute_pinn_loss(model, x_coll, t_coll, diff_coef):
    coef_tensor = torch.tensor(diff_coef, dtype=x_coll.dtype, device=x_coll.device)
    x_coll.requires_grad_(True)
    t_coll.requires_grad_(True)

    u = model(x_coll, t_coll)
    u_t = torch.autograd.grad(u, t_coll, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x_coll, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x_coll, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    loss_pde = torch.mean((u_t - coef_tensor * u_xx) ** 2)

    t0 = torch.zeros_like(x_coll)
    u_ic = model(x_coll, t0)
    u_ic_true = initial_gaussian(x_coll).unsqueeze(1)
    loss_ic = torch.mean((u_ic - u_ic_true) ** 2)

    x_left = -torch.ones_like(t_coll)
    x_right = torch.ones_like(t_coll)
    x_left.requires_grad_()
    x_right.requires_grad_()
    u_left = model(x_left, t_coll)
    u_right = model(x_right, t_coll)
    u_left_x = torch.autograd.grad(u_left, x_left, grad_outputs=torch.ones_like(u_left), create_graph=True)[0]
    u_right_x = torch.autograd.grad(u_right, x_right, grad_outputs=torch.ones_like(u_right), create_graph=True)[0]
    loss_bc = torch.mean(u_left_x ** 2) + torch.mean(u_right_x ** 2)

    return loss_pde + loss_ic + loss_bc

def train_pinn(model, diff_coef, num_epochs=200, lr=1e-3, device='cpu'):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    nx, nt = 100, 100
    x = torch.linspace(-1, 1, nx, device=device)
    t = torch.linspace(0, 1, nt, device=device)
    X, T = torch.meshgrid(x, t, indexing='ij')
    x_coll = X.flatten()
    t_coll = T.flatten()

    history = []
    for epoch in range(1, num_epochs+1):
        optimizer.zero_grad()
        loss = compute_pinn_loss(model, x_coll, t_coll, diff_coef)
        loss.backward()
        optimizer.step()
        history.append(loss.item())
        if epoch % 100 == 0:
            print(f"Epoch {epoch}/{num_epochs}, Loss={loss:.3e}")
    return history

def analytic_gaussian(x, t, c, sigma0=SIGMA0, diff=0.01, amp=1.0):
    sigma_t = np.sqrt(sigma0**2 + 2*diff*t)
    return amp * sigma0/sigma_t * np.exp(- (x-c)**2/(2*sigma_t**2))

def full_analytic(X, T, diff):
    centers = compute_centers(NUM_GAUSSIANS, GAUSSIAN_SPACING, torch.device('cpu')).cpu().numpy()
    U = np.zeros_like(X)
    for c in centers:
        U += analytic_gaussian(X, T, c, sigma0=SIGMA0, diff=diff)
    return U

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = PINN([2,64,64,64,1]).to(device)
    diff_coef = 0.01

    print("Training PINN with multiple Gaussians...")
    loss_hist = train_pinn(model, diff_coef, num_epochs=1000, lr=1e-3, device=device)

    nx_plot, nt_plot = 200, 200
    x_plot = np.linspace(-1,1,nx_plot)
    t_plot = np.linspace(0,1,nt_plot)
    Xp, Tp = np.meshgrid(x_plot, t_plot, indexing='ij')
    x_t = torch.tensor(Xp.flatten(), dtype=torch.float32, device=device)
    t_t = torch.tensor(Tp.flatten(), dtype=torch.float32, device=device)

    with torch.no_grad():
        U_pred = model(x_t, t_t).cpu().numpy().reshape(nx_plot, nt_plot)

    plt.figure(figsize=(8,6))
    plt.imshow(U_pred.T, extent=[-1,1,0,1], origin='lower', aspect='auto', cmap='jet')
    plt.colorbar(label='u(x,t)')
    plt.title('PINN Solution')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.tight_layout()
    plt.show()

    U_an = full_analytic(Xp, Tp, diff_coef)
    plt.figure(figsize=(8,6))
    plt.imshow(U_an.T, extent=[-1,1,0,1], origin='lower', aspect='auto', cmap='jet')
    plt.colorbar(label='u(x,t)')
    plt.title('Analytical Solution')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8,5))
    plt.plot(loss_hist)
    plt.yscale('log')
    plt.title('Learning Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
