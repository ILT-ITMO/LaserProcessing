import torch
import config
from normalizer import QuartzNormalizer
import physical_params as phys

# Используем нормализатор из config
normalizer = config.normalizer

def compute_centers(n, spacing, device):
    if n % 2 == 1:
        idx = torch.arange(-(n//2), n//2 + 1, device=device, dtype=torch.float32)
    else:
        half = n // 2
        idx = torch.arange(-half + 0.5, half, step=1.0, device=device, dtype=torch.float32)

    centers = []
    for i in idx:
        for j in idx:
            for k in idx:
                centers.append(torch.tensor([i * spacing, j * spacing, k * spacing], device=device))
    return centers

def initial_gaussian(x_tensor, y_tensor, z_tensor, t0=1.0):
    """Начальное условие в безразмерных координатах"""
    centers = compute_centers(phys.NUM_GAUSSIANS, phys.GAUSSIAN_SPACING, x_tensor.device)
    gaussians = []
    for c in centers:
        r_squared = (x_tensor - c[0])**2 + (y_tensor - c[1])**2 + (z_tensor - c[2])**2
        gaussians.append(t0 * torch.exp(-r_squared / (2 * phys.SIGMA0**2)))
    return sum(gaussians)

def laser_source_term(x_tensor, y_tensor, z_tensor, t_tensor, H, MU):
    """
    ПРАВИЛЬНАЯ функция лазерного источника тепла согласно физике
    Соответствует закону Бугера-Ламберта для экспоненциального поглощения
    """
    MU_STAR = H * MU

    # Центрируем координаты (пучок в центре области)
    x_centered = x_tensor - 0.5  
    y_centered = y_tensor - 0.5
    
    spatial_dist = torch.exp(-(x_centered**2 + y_centered**2) / (2 * phys.LASER_SIGMA**2))
    
    t_mod = torch.fmod(t_tensor, phys.LASER_PULSE_PERIOD)
    # Центр импульса в середине длительности
    pulse_center = phys.LASER_PULSE_DURATION / 2
    temporal_dist = torch.exp(-(t_mod - pulse_center)**2 / (2 * (phys.LASER_PULSE_DURATION/4)**2))
    
    depth_dist = MU_STAR * torch.exp(-MU_STAR * z_tensor)
    
    # Нормируем временной профиль чтобы интеграл = 1
    # temporal_norm = 1.0 / (config.LASER_PULSE_DURATION * torch.sqrt(torch.tensor(2 * torch.pi)))
    # temporal_dist = temporal_dist * temporal_norm
    
    return phys.I0 * spatial_dist * temporal_dist * depth_dist