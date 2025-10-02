import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import config
from conditions import initial_gaussian, laser_source_term, OpticalSource
import physical_params as phys

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

def compute_pinn_loss(model, source, x_coll: torch.Tensor, y_coll: torch.Tensor, z_coll: torch.Tensor, 
                     t_coll: torch.Tensor, material, c_p=phys.SPECIFIC_HEAT, k=phys.THERMAL_CONDUCTIVITY, rho=phys.DENSITY):

    x_coll.requires_grad_(True)
    y_coll.requires_grad_(True)
    z_coll.requires_grad_(True)
    t_coll.requires_grad_(True)

    # L_x, L_y, H_z, T_w = material
    # Fo_x = (k * T_w) / (rho * c_p * L_x**2) 
    # Fo_y = (k * T_w) / (rho * c_p * L_y**2)   
    # Fo_z = (k * T_w) / (rho * c_p * H_z*2)
    Fo_x, Fo_y, Fo_z = material

    u = model(x_coll, y_coll, z_coll, t_coll)
    u_t = torch.autograd.grad(u, t_coll, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x_coll, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x_coll, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_y = torch.autograd.grad(u, y_coll, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y_coll, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
    u_z = torch.autograd.grad(u, z_coll, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_zz = torch.autograd.grad(u_z, z_coll, grad_outputs=torch.ones_like(u_z), create_graph=True)[0]    
 
    source_term = source.forward(x_coll, y_coll, z_coll, t_coll)
    # source_term = (T_w / (rho * c_p * H_z)) * laser_source_term(x_coll, y_coll, z_coll, t_coll, H_z, phys.MU) # непонятно что делать с дельтой Т что это и как на нее домножить
    loss_pde = torch.mean((u_t - (Fo_x * u_xx + Fo_y * u_yy + Fo_z * u_zz) - source_term) ** 2)  

    # Начальное условие
    t0 = torch.zeros_like(x_coll)
    u_ic = model(x_coll, y_coll, z_coll, t0)
    # u_ic_true = initial_gaussian(x_coll, y_coll, z_coll).unsqueeze(1) # добавить нач темп и убрать гаус
    u_ic_true = torch.zeros_like(u_ic) * 20.0
    loss_ic = torch.mean((u_ic - u_ic_true) ** 2) 

    # УСЛОВИЯ НЕЙМАНА на всех границах (нулевой поток тепла)
    loss_bc = 0.0
    

    x_left = torch.full_like(t_coll, fill_value=config.X_MIN, device=x_coll.device).requires_grad_(True)
    x_right = torch.full_like(t_coll, fill_value=config.X_MAX, device=x_coll.device).requires_grad_(True)
    
    u_left = model(x_left, y_coll, z_coll, t_coll)
    u_right = model(x_right, y_coll, z_coll, t_coll)
    u_x_left = torch.autograd.grad(u_left, x_left, grad_outputs=torch.ones_like(u_left), create_graph=True)[0]
    u_x_right = torch.autograd.grad(u_right, x_right, grad_outputs=torch.ones_like(u_right), create_graph=True)[0]
    
    loss_bc += torch.mean(u_x_left ** 2) + torch.mean(u_x_right ** 2)
    

    y_bottom = torch.full_like(t_coll, fill_value=config.Y_MIN, device=x_coll.device).requires_grad_(True)
    y_top = torch.full_like(t_coll, fill_value=config.Y_MAX, device=x_coll.device).requires_grad_(True)
    
    u_bottom = model(x_coll, y_bottom, z_coll, t_coll)
    u_top = model(x_coll, y_top, z_coll, t_coll)
    u_y_bottom = torch.autograd.grad(u_bottom, y_bottom, grad_outputs=torch.ones_like(u_bottom), create_graph=True)[0]
    u_y_top = torch.autograd.grad(u_top, y_top, grad_outputs=torch.ones_like(u_top), create_graph=True)[0]
    
    loss_bc += torch.mean(u_y_bottom ** 2) + torch.mean(u_y_top ** 2)
    

    z_bottom = torch.full_like(t_coll, fill_value=config.Z_MIN, device=x_coll.device).requires_grad_(True)
    z_top = torch.full_like(t_coll, fill_value=config.Z_MAX, device=x_coll.device).requires_grad_(True)
    
    u_bottom_z = model(x_coll, y_coll, z_bottom, t_coll)
    u_top_z = model(x_coll, y_coll, z_top, t_coll)
    u_z_bottom = torch.autograd.grad(u_bottom_z, z_bottom, grad_outputs=torch.ones_like(u_bottom_z), create_graph=True)[0]
    u_z_top = torch.autograd.grad(u_top_z, z_top, grad_outputs=torch.ones_like(u_top_z), create_graph=True)[0]
    
    loss_bc += torch.mean(u_z_bottom ** 2) + torch.mean(u_z_top ** 2)


    w_pde, w_ic, w_bc = 1.0, 1.0, 1.0
    
    return {'loss':(w_pde * loss_pde + w_ic * loss_ic + w_bc * loss_bc),
            'loss_pde': loss_pde,
            'loss_bc': loss_bc,
            'loss_ic': loss_ic}

def train_pinn(model, source, material, num_epochs=200, lr=1e-3, device='cpu'):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Создаем сетку в безразмерных координатах [0,1]
    nx, ny, nz, nt = 20, 20, 20, 20
    x = torch.linspace(config.X_MIN, config.X_MAX, nx, device=device)
    y = torch.linspace(config.Y_MIN, config.Y_MAX, ny, device=device)
    z = torch.linspace(config.Z_MIN, config.Z_MAX, nz, device=device)
    t = torch.linspace(0, config.T_MAX, nt, device=device)
    
    X, Y, Z, T = torch.meshgrid(x, y, z, t, indexing='ij')
    x_coll = X.flatten()
    y_coll = Y.flatten()
    z_coll = Z.flatten()
    t_coll = T.flatten()

    history = []
    for epoch in tqdm(range(1, num_epochs+1)):
        optimizer.zero_grad()
        loss = compute_pinn_loss(model, source, x_coll, y_coll, z_coll, t_coll, material)
        loss['loss'].backward()
        optimizer.step()
        history.append(loss)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{num_epochs}, Loss={loss['loss']}")
            print(f"Loss_PDE={loss['loss_pde']} | Loss_BC={loss['loss_bc']} | Loss_IC={loss['loss_ic']}")
    return history