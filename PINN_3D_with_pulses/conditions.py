import torch
import config


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
    centers = compute_centers(config.NUM_GAUSSIANS, config.GAUSSIAN_SPACING, x_tensor.device)
    gaussians = []
    for c in centers:
        r_squared = (x_tensor - c[0])**2 + (y_tensor - c[1])**2 + (z_tensor - c[2])**2
        gaussians.append(t0 * torch.exp(-r_squared / (2 * config.SIGMA0**2)))

    return sum(gaussians)



def laser_source_term(x_tensor, y_tensor, z_tensor, t_tensor, amplitude=config.LASER_AMPLITUDE, 
                     pulse_duration=config.LASER_PULSE_DURATION, pulse_period=config.LASER_PULSE_PERIOD,
                     sigma=config.LASER_SIGMA):
    """
    Функция лазерного источника тепла
    """

    spatial_dist = amplitude * torch.exp(-(x_tensor**2 + y_tensor**2) / (2 * sigma**2))
    
    t_mod = torch.fmod(t_tensor, pulse_period)
    temporal_dist = torch.where(
        (t_mod >= 0) & (t_mod <= pulse_duration),
        torch.ones_like(t_tensor),
        torch.zeros_like(t_tensor)
    )
    
    z_dist = torch.exp(-z_tensor**2 / (2 * (sigma/2)**2)) 
    
    return spatial_dist * temporal_dist * z_dist