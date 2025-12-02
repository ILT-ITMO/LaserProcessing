import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from conditions import laser_source_term
import config

class PINN(nn.Module):
    def __init__(self, layers_sizes):
        super().__init__()
        layers = []
        for i in range(len(layers_sizes) - 1):
            layers.append(nn.Linear(layers_sizes[i], layers_sizes[i + 1]))
            if i < len(layers_sizes) - 2:
                layers.append(nn.Tanh())
        self.network = nn.Sequential(*layers)

    def forward(self, x_tensor, y_tensor, z_tensor, t_tensor):
        inputs = torch.stack([x_tensor, y_tensor, z_tensor, t_tensor], dim=1)
        return self.network(inputs)


def compute_pinn_loss(model, x_coll: torch.Tensor, y_coll: torch.Tensor, z_coll: torch.Tensor, 
                     t_coll: torch.Tensor, diff_coef, laser_mode=None):
    """
    Вычисление loss для PINN с возможностью выбора режима лазера
    
    Args:
        laser_mode: режим лазера ("pulsed" или "continuous"). Если None, берется из config
    """
    coef_tensor = torch.tensor(diff_coef, dtype=x_coll.dtype, device=x_coll.device)
    
    x_coll.requires_grad_(True)
    y_coll.requires_grad_(True)
    z_coll.requires_grad_(True)
    t_coll.requires_grad_(True)

    u = model(x_coll, y_coll, z_coll, t_coll)
    
    u_t = torch.autograd.grad(u, t_coll, grad_outputs=torch.ones_like(u), create_graph=True)[0]

    u_x = torch.autograd.grad(u, x_coll, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x_coll, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    
    u_y = torch.autograd.grad(u, y_coll, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y_coll, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
    
    u_z = torch.autograd.grad(u, z_coll, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_zz = torch.autograd.grad(u_z, z_coll, grad_outputs=torch.ones_like(u_z), create_graph=True)[0]    
    
    # Источник тепла с учетом режима лазера
    source_term = laser_source_term(x_coll, y_coll, z_coll, t_coll, laser_mode=laser_mode)
    
    # Безразмерное уравнение теплопроводности
    loss_pde = torch.mean((u_t - coef_tensor * (u_xx + u_yy + u_zz) - source_term) ** 2)

    # Начальное условие
    t0 = torch.zeros_like(x_coll)
    u_ic = model(x_coll, y_coll, z_coll, t0)
    u_ic_true = torch.zeros_like(u_ic)
    loss_ic = torch.mean((u_ic - u_ic_true) ** 2)

    loss_bc = 0.0
    # Граничные условия Неймана (нулевой поток) на всех границах
    
    # Границы по x (левая и правая)
    x_left = -torch.ones_like(x_coll).requires_grad_(True)
    x_right = torch.ones_like(x_coll).requires_grad_(True)
    
    u_left = model(x_left, y_coll, z_coll, t_coll)
    u_right = model(x_right, y_coll, z_coll, t_coll)
    
    u_left_x = torch.autograd.grad(u_left, x_left, 
                                 grad_outputs=torch.ones_like(u_left), 
                                 create_graph=True, retain_graph=True)[0]
    u_right_x = torch.autograd.grad(u_right, x_right, 
                                  grad_outputs=torch.ones_like(u_right), 
                                  create_graph=True, retain_graph=True)[0]
    
    loss_bc += torch.mean(u_left_x ** 2) + torch.mean(u_right_x ** 2)  # ∂u/∂x = 0
    
    # Границы по y (нижняя и верхняя)
    y_bottom = -torch.ones_like(y_coll).requires_grad_(True)
    y_top = torch.ones_like(y_coll).requires_grad_(True)
    
    u_bottom = model(x_coll, y_bottom, z_coll, t_coll)
    u_top = model(x_coll, y_top, z_coll, t_coll)
    
    u_bottom_y = torch.autograd.grad(u_bottom, y_bottom, 
                                   grad_outputs=torch.ones_like(u_bottom), 
                                   create_graph=True, retain_graph=True)[0]
    u_top_y = torch.autograd.grad(u_top, y_top, 
                                grad_outputs=torch.ones_like(u_top), 
                                create_graph=True, retain_graph=True)[0]
    
    loss_bc += torch.mean(u_bottom_y ** 2) + torch.mean(u_top_y ** 2)  # ∂u/∂y = 0
    
    # Границы по z (нижняя и верхняя)
    z_bottom = torch.zeros_like(z_coll).requires_grad_(True)
    z_top = torch.ones_like(z_coll).requires_grad_(True)
    
    u_bottom_z = model(x_coll, y_coll, z_bottom, t_coll)
    u_top_z = model(x_coll, y_coll, z_top, t_coll)
    
    u_bottom_z_deriv = torch.autograd.grad(u_bottom_z, z_bottom, 
                                         grad_outputs=torch.ones_like(u_bottom_z), 
                                         create_graph=True, retain_graph=True)[0]
    u_top_z_deriv = torch.autograd.grad(u_top_z, z_top, 
                                      grad_outputs=torch.ones_like(u_top_z), 
                                      create_graph=True, retain_graph=True)[0]
    
    loss_bc += torch.mean(u_bottom_z_deriv ** 2) + torch.mean(u_top_z_deriv ** 2)  # ∂u/∂z = 0

    w_pde, w_ic, w_bc = 1.0, 1.0, 2.0
    
    return w_pde * loss_pde + w_ic * loss_ic + w_bc * loss_bc

def train_pinn(model, diff_coef, num_epochs=200, lr=1e-3, device='cpu', laser_mode=None):
    """
    Обучение PINN с возможностью выбора режима лазера
    
    Args:
        laser_mode: режим лазера ("pulsed" или "continuous"). Если None, берется из config
    """
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    if laser_mode is None:
        laser_mode = config.LASER_MODE
    
    # Область в безразмерных координатах
    nx, ny, nz, nt = 20, 20, 20, 20
    x = torch.linspace(-1, 1, nx, device=device)  # ±1 радиус пучка
    y = torch.linspace(-1, 1, ny, device=device)  # ±1 радиус пучка  
    z = torch.linspace(0, 1, nz, device=device) 
    t = torch.linspace(0, config.SIMULATION_TIME_NORM, nt, device=device)  
    
    X, Y, Z, T = torch.meshgrid(x, y, z, t, indexing='ij')
    x_coll = X.flatten()
    y_coll = Y.flatten()
    z_coll = Z.flatten()
    t_coll = T.flatten()

    history = []
    for epoch in tqdm(range(1, num_epochs+1)):
        optimizer.zero_grad()
        loss = compute_pinn_loss(model, x_coll, y_coll, z_coll, t_coll, 
                                diff_coef, laser_mode=laser_mode)
        loss.backward()
        optimizer.step()
        history.append(loss.item())
        if epoch % 100 == 0:
            print(f"Epoch {epoch}/{num_epochs}, Loss={loss:.3e}, Mode={laser_mode}")
    return history