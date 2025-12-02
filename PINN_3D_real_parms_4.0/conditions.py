import torch
import config
import math

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
    centers = compute_centers(config.NUM_GAUSSIANS, config.GAUSSIAN_SPACING, x_tensor.device)
    gaussians = []
    for c in centers:
        r_squared = (x_tensor - c[0])**2 + (y_tensor - c[1])**2 + (z_tensor - c[2])**2
        gaussians.append(t0 * torch.exp(-r_squared / (2 * config.SIGMA0**2)))

    return sum(gaussians)

# def laser_source_term(x_tensor, y_tensor, z_tensor, t_tensor, 
#                      amplitude=config.LASER_AMPLITUDE,
#                      pulse_sigma=config.LASER_PULSE_SIGMA_NORM,
#                      pulse_period=config.LASER_PULSE_PERIOD_NORM,
#                      beam_sigma=config.LASER_SIGMA_NORM):
#     """
#     Функция лазерного источника тепла в безразмерных координатах
#     с ГАУССОВЫМ временным профилем
#     """
    
#     # Безразмерные координаты (нормированы на радиус пучка)
#     # x_norm, y_norm уже безразмерные
#     # z_norm = z_physical / CHARACTERISTIC_LENGTH
    
#     # 1. Пространственное распределение (движущийся гауссов пучок)
#     scan_velocity_norm = config.LASER_SCAN_VELOCITY * config.CHARACTERISTIC_TIME / config.CHARACTERISTIC_LENGTH
#     x_center = scan_velocity_norm * t_tensor  # Сканирование вдоль оси X
    
#     r2 = (x_tensor - x_center)**2 + y_tensor**2
#     spatial_dist = amplitude * torch.exp(-2 * r2 / (beam_sigma**2))
    
#     # 2. Временное распределение (ГАУССОВОЕ вместо прямоугольного)
#     t_mod = torch.fmod(t_tensor, pulse_period)
#     # Гауссов импульс с центром в середине периода импульса

#     # temporal_dist = torch.exp(-(t_mod - pulse_sigma * 3)**2 / (2 * pulse_sigma**2))
#     temporal_dist = torch.exp(-(t_mod - pulse_period/2)**2 / (2 * pulse_sigma**2))

#     temporal_dist = (1.0 / (pulse_sigma * math.sqrt(2 * math.pi))) * torch.exp(-(t_mod - pulse_period/2)**2 / (2 * pulse_sigma**2))

#     # 3. Глубинное распределение (экспоненциальное поглощение по Бугеру-Ламберту)
#     # Безразмерный коэффициент поглощения
#     alpha_norm = config.MATERIAL_ABSORPTION * config.CHARACTERISTIC_LENGTH
#     z_dist = torch.exp(-alpha_norm * z_tensor)
    
#     return spatial_dist * temporal_dist * z_dist

# def laser_source_term(x_tensor, y_tensor, z_tensor, t_tensor, 
#                      amplitude=config.LASER_AMPLITUDE,
#                      pulse_sigma=config.LASER_PULSE_SIGMA_NORM,
#                      pulse_period=config.LASER_PULSE_PERIOD_NORM,
#                      beam_sigma=config.LASER_SIGMA_NORM):
#     """
#     Функция лазерного источника тепла в безразмерных координатах
#     с ГАУССОВЫМ временным профилем дала хорошую температуру порядка 3000 К
#     """
    
#     # Пространственное распределение (движущийся гауссов пучок)
#     scan_velocity_norm = config.LASER_SCAN_VELOCITY * config.CHARACTERISTIC_TIME / config.CHARACTERISTIC_LENGTH
#     x_center = scan_velocity_norm * t_tensor  # Сканирование вдоль оси X
    
#     r2 = (x_tensor - x_center)**2 + y_tensor**2
#     spatial_dist = amplitude * torch.exp(-2 * r2 / (beam_sigma**2))
    
#     # ИСПРАВЛЕННОЕ временное распределение (ГАУССОВОЕ)
#     t_mod = torch.fmod(t_tensor, pulse_period)
#     # Центр импульса в середине периода
#     pulse_center = pulse_period / 2
#     temporal_dist = torch.exp(-(t_mod - pulse_center)**2 / (2 * pulse_sigma**2))
    
#     # Нормализация чтобы интеграл = 1 (опционально)
#     # temporal_dist = temporal_dist / (pulse_sigma * math.sqrt(2 * math.pi))
    
#     # Глубинное распределение (экспоненциальное поглощение по Бугеру-Ламберту)
#     alpha_norm = config.MATERIAL_ABSORPTION * config.CHARACTERISTIC_LENGTH
#     z_dist = torch.exp(-alpha_norm * z_tensor)
    
#     return spatial_dist * temporal_dist * z_dist


def laser_source_term(x_tensor, y_tensor, z_tensor, t_tensor, 
                     amplitude=config.LASER_AMPLITUDE,
                     pulse_sigma=config.LASER_PULSE_SIGMA_NORM,
                     pulse_period=config.LASER_PULSE_PERIOD_NORM,
                     beam_sigma=config.LASER_SIGMA_NORM):
    """
    Функция лазерного источника тепла в безразмерных координатах
    с ГАУССОВЫМ временным профилем и СТАТИЧНЫМ пучком
    """
    
    # ПРОСТРАНСТВЕННОЕ РАСПРЕДЕЛЕНИЕ - СТАТИЧНЫЙ пучок в центре (0,0)
    r2 = x_tensor**2 + y_tensor**2  # центр всегда в (0,0)
    spatial_dist = amplitude * torch.exp(-r2 / (beam_sigma**2))
    
    # ВРЕМЕННОЕ РАСПРЕДЕЛЕНИЕ - ГАУССОВ импульс
    t_mod = torch.fmod(t_tensor, pulse_period)
    
    # Центр импульса точно в середине периода
    pulse_center = pulse_period / 2.0
    
    # Гауссов импульс с правильным центром
    temporal_dist = torch.exp(-(t_mod - pulse_center)**2 / (2 * pulse_sigma**2))
    
    # ГЛУБИННОЕ РАСПРЕДЕЛЕНИЕ
    alpha_norm = config.MATERIAL_ABSORPTION * config.CHARACTERISTIC_LENGTH
    z_dist = torch.exp(-alpha_norm * z_tensor)
    
    source = spatial_dist * temporal_dist * z_dist
    
    return source



def convert_to_physical_coords(x_norm, y_norm, z_norm, t_norm):
    """Конвертация безразмерных координат в физические"""
    x_phys = x_norm * config.CHARACTERISTIC_LENGTH
    y_phys = y_norm * config.CHARACTERISTIC_LENGTH  
    z_phys = z_norm * config.CHARACTERISTIC_LENGTH
    t_phys = t_norm * config.CHARACTERISTIC_TIME
    return x_phys, y_phys, z_phys, t_phys

def convert_to_physical_temperature(T_norm):
    """Конвертация безразмерной температуры в физическую"""
    return config.INITIAL_TEMPERATURE + T_norm * config.CHARACTERISTIC_TEMPERATURE

def get_physical_extent(x_norm_range, y_norm_range, z_norm_range):
    """Получить физические границы для визуализации"""
    x_phys_min = x_norm_range[0] * config.CHARACTERISTIC_LENGTH * 1e6  # в мкм
    x_phys_max = x_norm_range[1] * config.CHARACTERISTIC_LENGTH * 1e6
    y_phys_min = y_norm_range[0] * config.CHARACTERISTIC_LENGTH * 1e6
    y_phys_max = y_norm_range[1] * config.CHARACTERISTIC_LENGTH * 1e6
    z_phys_min = z_norm_range[0] * config.CHARACTERISTIC_LENGTH * 1e6
    z_phys_max = z_norm_range[1] * config.CHARACTERISTIC_LENGTH * 1e6
    
    return (x_phys_min, x_phys_max), (y_phys_min, y_phys_max), (z_phys_min, z_phys_max)