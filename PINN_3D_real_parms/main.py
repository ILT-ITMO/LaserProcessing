# main.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from visual import visualize_laser_pulses, visualize_laser_spatial_profile, create_animation
from pinn import PINN, train_pinn
import config
from normalizer import QuartzNormalizer
import physical_params

def postprocess_results(model, device='cpu'):
    """Постобработка результатов с преобразованием в размерные величины"""

    x_phys = np.linspace(physical_params.X_MIN, physical_params.X_MAX, 30)  # м
    y_phys = np.linspace(physical_params.Y_MIN, physical_params.Y_MAX, 30)  # м
    z_phys = np.linspace(physical_params.Z_MIN, physical_params.Z_MAX, 20)  # м
    t_phys = np.linspace(0, physical_params.T_MAX, 50)           # с
    

    # (x_norm, x_coef), (y_norm, y_coef), (z_norm, z_coef), (t_norm, t_coef) = QuartzNormalizer.normalize_vector(x_phys, y_phys, z_phys, t_phys, get_coef=True)
    
    with torch.no_grad():
        Xp, Yp, Zp, Tp = np.meshgrid(x_norm.numpy(), y_norm.numpy(), 
                                    z_norm.numpy(), t_norm.numpy(), indexing='ij')
        x_t = torch.tensor(Xp.flatten(), dtype=torch.float32, device=device)
        y_t = torch.tensor(Yp.flatten(), dtype=torch.float32, device=device)
        z_t = torch.tensor(Zp.flatten(), dtype=torch.float32, device=device)
        t_t = torch.tensor(Tp.flatten(), dtype=torch.float32, device=device)
        
        u_pred_dimless = model(x_t, y_t, z_t, t_t).cpu().numpy()
    
    # Преобразуем обратно в размерные величины
    u_pred_dimensional = u_pred_dimless * normalizer.T_scale  # ТУТ нужно переписать код !!!!!!
    
    return {
        'x_physical': x_phys,
        'y_physical': y_phys, 
        'z_physical': z_phys,
        't_physical': t_phys,
        'temperature': u_pred_dimensional.reshape(len(x_phys), len(y_phys), 
                                                len(z_phys), len(t_phys))
    }

if __name__ == '__main__':
    try:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    except:
        device = torch.device("cuda" if torch.backends.cuda.is_built() else "cpu")

    print(f"Using device: {device}")

    os.makedirs('animations', exist_ok=True)

    # Используем нормализатор из config
    # normalizer = config.normalizer

    print("Визуализация профиля лазерных импульсов...")
    visualize_laser_pulses()
    visualize_laser_spatial_profile()

    model = PINN([4, 128, 128, 128, 1]).to(device)
    
    # Используем безразмерный коэффициент диффузии из config
    # diff_coef = config.DIFF_COEF

    x_phys = np.linspace(physical_params.X_MIN, physical_params.X_MAX, 20)  # м
    y_phys = np.linspace(physical_params.Y_MIN, physical_params.Y_MAX, 20)  # м
    z_phys = np.linspace(physical_params.Z_MIN, physical_params.Z_MAX, 20)  # м
    t_phys = np.linspace(0, physical_params.T_MAX, 20) 

    (x_norm, x_coef), (y_norm, y_coef), (z_norm, z_coef), (t_norm, t_coef) = QuartzNormalizer.normalize_vector(x_phys, y_phys, z_phys, t_phys, get_coef=True)

    print("Training 3D PINN with laser source...")
    material = x_coef, y_coef, z_coef, t_coef
    loss_hist = train_pinn(model, material, num_epochs=1000, lr=1e-3, device=device)

    # Постобработка результатов
    print("Postprocessing results...")
    results = postprocess_results(model, device)
    
    # Создаем анимацию с размерными величинами
    print("Creating animations with dimensional units...")
    
    plt.figure(figsize=(8,5))
    plt.plot(loss_hist)
    plt.yscale('log')
    plt.title('Learning Curve (Quartz JGS1 Laser Heating)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('animations/learning_curve_quartz.png')
    plt.show()

    print("Все анимации сохранены в папке 'animations/'")