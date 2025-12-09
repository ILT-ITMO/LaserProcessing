import torch
import config
import math

def compute_centers(n, spacing, device):
    """
    Computes the centers of a 3D grid with specified spacing.
    
    This method generates coordinates for the center of each point in a 3D grid.
    The grid is centered at the origin by adjusting the index range based on
    whether the grid size is odd or even. This allows for accurate representation
    of points in 3D space for subsequent calculations or simulations.
    
    Args:
      n: The size of the grid along each dimension (integer).
      spacing: The distance between grid points (float).
      device: The device to use for tensor creation (e.g., 'cpu', 'cuda').
    
    Returns:
      A list of torch.Tensor objects, where each tensor contains the
      (x, y, z) coordinates of a grid center.
    """
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
    """
    Initializes a Gaussian distribution as the initial condition for simulating heat distribution.
    
    Args:
        x_tensor (torch.Tensor): Tensor representing the x-coordinates of the spatial domain.
        y_tensor (torch.Tensor): Tensor representing the y-coordinates of the spatial domain.
        z_tensor (torch.Tensor): Tensor representing the z-coordinates of the spatial domain.
        t0 (float, optional): Initial amplitude of the Gaussian. Defaults to 1.0.
    
    Returns:
        torch.Tensor: A tensor representing the initial heat distribution, modeled as a sum of Gaussian functions.
    
    This method creates an initial heat distribution by placing Gaussian 'hotspots' at several points in space. These hotspots serve as the starting point for simulating how heat diffuses through the material when exposed to a laser. The positions of these hotspots are determined by `compute_centers`, and their initial intensity is defined by `t0` and `config.SIGMA0`.
    """
    centers = compute_centers(config.NUM_GAUSSIANS, config.GAUSSIAN_SPACING, x_tensor.device)
    gaussians = []
    for c in centers:
        r_squared = (x_tensor - c[0])**2 + (y_tensor - c[1])**2 + (z_tensor - c[2])**2
        gaussians.append(t0 * torch.exp(-r_squared / (2 * config.SIGMA0**2)))

    return sum(gaussians)

def laser_source_term(x_tensor, y_tensor, z_tensor, t_tensor, 
                     amplitude=None,
                     pulse_sigma=None,
                     pulse_period=None,
                     beam_sigma=None,
                     laser_mode=None):
    """
    Функция лазерного источника тепла в безразмерных координатах.
    Определяет распределение тепла, генерируемого лазером, в пространстве и времени.
    Поддерживает два режима: импульсный (pulsed) и непрерывный (continuous).
    
    Args:
        x_tensor (torch.Tensor): Тензор x-координат.
        y_tensor (torch.Tensor): Тензор y-координат.
        z_tensor (torch.Tensor): Тензор z-координат.
        t_tensor (torch.Tensor): Тензор времени.
        amplitude (float, optional): Амплитуда лазерного излучения. По умолчанию берется из config.
        pulse_sigma (float, optional): Ширина импульса. По умолчанию берется из config.
        pulse_period (float, optional): Период импульсов. По умолчанию берется из config.
        beam_sigma (float, optional): Ширина лазерного пучка. По умолчанию берется из config.
        laser_mode (str, optional): Режим лазера ("pulsed" или "continuous"). Если None, берется из config.
    
    Returns:
        torch.Tensor: Тензор, представляющий распределение источника тепла.
    """
    if laser_mode is None:
        laser_mode = config.LASER_MODE
    
    # Используем параметры по умолчанию если не указаны
    if amplitude is None:
        amplitude = config.LASER_AMPLITUDE
    if pulse_sigma is None:
        pulse_sigma = config.LASER_PULSE_SIGMA_NORM
    if pulse_period is None:
        pulse_period = config.LASER_PULSE_PERIOD_NORM
    if beam_sigma is None:
        beam_sigma = config.LASER_SIGMA_NORM
    
    # Пространственное распределение - СТАТИЧНЫЙ пучок в центре (0,0)
    r2 = x_tensor**2 + y_tensor**2  # центр всегда в (0,0)
    spatial_dist = amplitude * torch.exp(-r2 / (beam_sigma**2))
    
    # Временное распределение в зависимости от режима
    if laser_mode == "continuous":
        # Непрерывный режим - постоянный источник
        temporal_dist = torch.ones_like(t_tensor)
    else:
        # Импульсный режим - гауссовы импульсы
        if pulse_period > 0:
            t_mod = torch.fmod(t_tensor, pulse_period)
        else:
            t_mod = t_tensor
        pulse_center = pulse_period / 2.0 if pulse_period > 0 else 0.5
        temporal_dist = torch.exp(-(t_mod - pulse_center)**2 / (2 * pulse_sigma**2))
    
    # Глубинное распределение (экспоненциальное поглощение по Бугеру-Ламберту)
    alpha_norm = config.MATERIAL_ABSORPTION * config.CHARACTERISTIC_LENGTH
    z_dist = torch.exp(-alpha_norm * z_tensor)
    
    source = spatial_dist * temporal_dist * z_dist
    
    return source

def convert_to_physical_coords(x_norm, y_norm, z_norm, t_norm):
    """
    Converts normalized coordinates to physical coordinates.
    
    Args:
        x_norm (float): Normalized x-coordinate.
        y_norm (float): Normalized y-coordinate.
        z_norm (float): Normalized z-coordinate.
        t_norm (float): Normalized time.
    
    Returns:
        tuple: A tuple containing the physical x, y, z coordinates and time.
               (x_phys, y_phys, z_phys, t_phys)
    """
    x_phys = x_norm * config.CHARACTERISTIC_LENGTH
    y_phys = y_norm * config.CHARACTERISTIC_LENGTH  
    z_phys = z_norm * config.CHARACTERISTIC_LENGTH
    t_phys = t_norm * config.CHARACTERISTIC_TIME
    return x_phys, y_phys, z_phys, t_phys

def convert_to_physical_temperature(T_norm):
    """
    Converts a dimensionless temperature to a physical temperature.
    
    Args:
        T_norm (float): The dimensionless temperature value.
    
    Returns:
        float: The corresponding physical temperature.
    
    This method calculates the physical temperature based on a given dimensionless temperature, a base temperature, and a characteristic temperature. This conversion is essential for interpreting temperature values within the context of laser-material interaction simulations, allowing for meaningful analysis of thermal behavior.
    """
    return config.INITIAL_TEMPERATURE + T_norm * config.CHARACTERISTIC_TEMPERATURE

def get_physical_extent(x_norm_range, y_norm_range, z_norm_range):
    """
    Получить физические границы для визуализации.
    
    Args:
        x_norm_range (tuple): Normalized range of x-coordinates.
        y_norm_range (tuple): Normalized range of y-coordinates.
        z_norm_range (tuple): Normalized range of z-coordinates.
    
    Returns:
        tuple: A tuple containing the physical extents for x, y, and z dimensions, represented as tuples of (min, max) values in micrometers.
    
    The method converts normalized coordinates to physical dimensions based on a characteristic length, enabling accurate representation of the simulated or measured space. This conversion is crucial for relating the simulation results to real-world scales and for proper visualization of the data.
    """
    x_phys_min = x_norm_range[0] * config.CHARACTERISTIC_LENGTH * 1e6  # в мкм
    x_phys_max = x_norm_range[1] * config.CHARACTERISTIC_LENGTH * 1e6
    y_phys_min = y_norm_range[0] * config.CHARACTERISTIC_LENGTH * 1e6
    y_phys_max = y_norm_range[1] * config.CHARACTERISTIC_LENGTH * 1e6
    z_phys_min = z_norm_range[0] * config.CHARACTERISTIC_LENGTH * 1e6
    z_phys_max = z_norm_range[1] * config.CHARACTERISTIC_LENGTH * 1e6
    
    return (x_phys_min, x_phys_max), (y_phys_min, y_phys_max), (z_phys_min, z_phys_max)

def get_laser_parameters_for_mode(laser_mode=None):
    """
    Returns the laser parameters for a specified mode.
    
    Args:
        laser_mode (str, optional): The desired laser mode ("continuous" or "pulsed"). 
                                     If None, the default mode from the configuration is used. Defaults to None.
    
    Returns:
        dict: A dictionary containing the laser parameters for the given mode. 
              The dictionary includes parameters like mode, amplitude, pulse characteristics, 
              beam characteristics, peak power, and a descriptive string. 
              The parameters are sourced from the project's configuration.
    """
    if laser_mode is None:
        laser_mode = config.LASER_MODE
    
    if laser_mode == "continuous":
        return {
            "mode": "continuous",
            "amplitude": config.LASER_AMPLITUDE,
            "pulse_sigma": config.LASER_PULSE_SIGMA_NORM,
            "pulse_period": config.LASER_PULSE_PERIOD_NORM,
            "beam_sigma": config.LASER_SIGMA_NORM,
            "peak_power": config.LASER_CONTINUOUS_POWER,
            "description": "Непрерывный режим"
        }
    else:
        return {
            "mode": "pulsed",
            "amplitude": config.LASER_AMPLITUDE,
            "pulse_sigma": config.LASER_PULSE_SIGMA_NORM,
            "pulse_period": config.LASER_PULSE_PERIOD_NORM,
            "beam_sigma": config.LASER_SIGMA_NORM,
            "num_pulses": config.NUM_PULSES,
            "peak_power": config.LASER_PEAK_POWER,
            "description": f"Импульсный режим: {config.NUM_PULSES} импульсов"
        }