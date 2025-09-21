import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from visual import visualize_laser_pulses, visualize_laser_spatial_profile, create_animation
from pinn import PINN, train_pinn





if __name__ == '__main__':
    try:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    except:
        device = torch.device("cuda" if torch.backends.cuda.is_built() else "cpu")

    print(f"Using device: {device}")

    os.makedirs('animations', exist_ok=True)

    print("Визуализация профиля лазерных импульсов...")
    visualize_laser_pulses()
    visualize_laser_spatial_profile()

    model = PINN([4, 128, 128, 128, 1]).to(device)
    diff_coef = 0.01

    print("Training 3D PINN with laser source...")
    loss_hist = train_pinn(model, diff_coef, num_epochs=1000, lr=1e-3, device=device)

    nx_plot, ny_plot, nz_plot, nt_plot = 30, 30, 30, 20
    x_plot = np.linspace(-1, 1, nx_plot)
    y_plot = np.linspace(-1, 1, ny_plot)
    z_plot = np.linspace(0, 1, nz_plot)
    t_plot = np.linspace(0, 1, nt_plot)
    
    print("Making PINN predictions...")
    with torch.no_grad():
        Xp, Yp, Zp, Tp = np.meshgrid(x_plot, y_plot, z_plot, t_plot, indexing='ij')
        x_t = torch.tensor(Xp.flatten(), dtype=torch.float32, device=device)
        y_t = torch.tensor(Yp.flatten(), dtype=torch.float32, device=device)
        z_t = torch.tensor(Zp.flatten(), dtype=torch.float32, device=device)
        t_t = torch.tensor(Tp.flatten(), dtype=torch.float32, device=device)
        
        U_pred = model(x_t, y_t, z_t, t_t).cpu().numpy().reshape(nx_plot, ny_plot, nz_plot, nt_plot)


    print("Creating GIF animations with laser info...")
    create_animation(U_pred, x_plot, y_plot, z_plot, t_plot, 
                                   'PINN Solution with Laser Pulses', 
                                   'animations/pinn_solution_laser.gif')


    plt.figure(figsize=(8,5))
    plt.plot(loss_hist)
    plt.yscale('log')
    plt.title('Learning Curve (with Laser Source)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('animations/learning_curve_laser.png')
    plt.show()

    print("Все анимации сохранены в папке 'animations/'")