import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from visual import visualize_laser_pulses, visualize_laser_spatial_profile, create_animation
from pinn import PINN, train_pinn
from conditions import convert_to_physical_coords, convert_to_physical_temperature
import config

if __name__ == '__main__':
    try:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    except:
        device = torch.device("cuda" if torch.backends.cuda.is_built() else "cpu")

    print(f"Using device: {device}")

    os.makedirs('animations', exist_ok=True)

    print("Физические параметры системы:")
    print(f"Лазер: {config.LASER_AVG_POWER} Вт, {config.LASER_WAVELENGTH*1e6} мкм")
    print(f"Пучок: {config.LASER_BEAM_RADIUS*1e6} мкм, СТАТИЧНЫЙ")
    print(f"Материал: кварц JS1, поглощение: {config.MATERIAL_ABSORPTION} 1/м")
    print(f"Импульсы: Гауссовы, {config.LASER_PULSE_DURATION*1e6:.1f} мкс FWHM")
    print(f"Количество импульсов: {config.NUM_PULSES}")
    print(f"Общее время моделирования: {config.SIMULATION_TIME_NORM * config.CHARACTERISTIC_TIME*1e6:.1f} мкс")


    print("Визуализация профиля лазерных импульсов...")
    visualize_laser_pulses()
    visualize_laser_spatial_profile()

    model = PINN([4, 128, 128, 128, 1]).to(device)
    diff_coef = 1.0  # Безразмерный коэффициент = 1

    print("Training 3D PINN with physical laser source (Гауссовы импульсы)...")
    loss_hist = train_pinn(model, diff_coef, num_epochs=1000, lr=1e-3, device=device)

    nx_plot, ny_plot, nz_plot, nt_plot = 30, 30, 30, 20
    x_plot = np.linspace(-1, 1, nx_plot)  
    y_plot = np.linspace(-1, 1, ny_plot)  
    z_plot = np.linspace(0, 1, nz_plot)   
    t_plot = np.linspace(0, config.SIMULATION_TIME_NORM, nt_plot)  
    
    print("Making PINN predictions...")
    with torch.no_grad():
        Xp, Yp, Zp, Tp = np.meshgrid(x_plot, y_plot, z_plot, t_plot, indexing='ij')
        x_t = torch.tensor(Xp.flatten(), dtype=torch.float32, device=device)
        y_t = torch.tensor(Yp.flatten(), dtype=torch.float32, device=device)
        z_t = torch.tensor(Zp.flatten(), dtype=torch.float32, device=device)
        t_t = torch.tensor(Tp.flatten(), dtype=torch.float32, device=device)
        
        U_pred_norm = model(x_t, y_t, z_t, t_t).cpu().numpy().reshape(nx_plot, ny_plot, nz_plot, nt_plot)
        
        # Конвертация в физические величины
        U_pred_physical = convert_to_physical_temperature(U_pred_norm)

    print("Creating GIF animations with physical parameters...")
    create_animation(U_pred_norm, x_plot, y_plot, z_plot, t_plot, 
                                'PINN Solution: Нагрев кварца лазером\nСТАТИЧНЫЙ пучок (8 импульсов)', 
                                'animations/pinn_solution_8_pulses.gif')

    plt.figure(figsize=(8,5))
    plt.plot(loss_hist)
    plt.yscale('log')
    plt.title('Learning Curve (Physical Laser with Gaussian Pulses)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('animations/learning_curve_physical_gaussian.png')
    plt.show()

    print("Все анимации сохранены в папке 'animations/'")
    print(f"Максимальная температура: {np.max(U_pred_physical):.1f} K")
    print(f"Перегрев: {np.max(U_pred_physical) - config.INITIAL_TEMPERATURE:.1f} K")
    print(f"Физическое время моделирования: {config.CHARACTERISTIC_TIME*1e3:.1f} мс")