import torch
import numpy as np
import physical_params as phys

class QuartzNormalizer:
    """
    Класс для нормировки физических величин для кварца JGS1 и лазерного нагрева
    """
    
    def __init__(self):
        pass

        # self.L_char = max(phys.X_MAX - phys.X_MIN, phys.Y_MAX - phys.Y_MIN) / 2  # радиус области
        # self.H_char = phys.Z_MAX - phys.Z_MIN  # толщина образца
        
        # self.T_char = phys.LASER_PULSE_PERIOD  # 110 мкс
        
        # # ПРАВИЛЬНЫЙ расчет температурного масштаба через интенсивность лазера
        # # Интенсивность на поверхности: I0 = P / (π * w0^2)
        # laser_power = phys.LASER_AMPLITUDE * (np.pi * phys.LASER_SIGMA**2) * phys.LASER_PULSE_DURATION
        # I0 = laser_power / (np.pi * phys.LASER_SIGMA**2)
        
        # # Масштаб температуры: ΔT = (I0 * (1 - exp(-μ_*))) / (ρ * c * H / T_char)
        # absorbed_power = I0 * (1 - np.exp(-phys.MU_STAR))
        # self.T_scale = (absorbed_power * self.T_char) / (phys.DENSITY * phys.SPECIFIC_HEAT * self.H_char)
        
        # # Масштаб источника тепла (УПРОСТИТЬ)
        # self.Q_scale = (phys.DENSITY * phys.SPECIFIC_HEAT * self.T_scale) / self.T_char
        
        # # Безразмерные параметры (ДОБАВИТЬ mu_star)
        # self.alpha_star = (phys.THERMAL_DIFFUSIVITY * self.T_char) / (self.L_char**2)
        # self.mu_star = phys.MU_STAR  # ← ДОБАВИТЬ
        # self.laser_amplitude_star = phys.LASER_AMPLITUDE / self.Q_scale
        # self.laser_sigma_star = phys.LASER_SIGMA / self.L_char
        # self.pulse_duration_star = phys.LASER_PULSE_DURATION / self.T_char
        # self.pulse_period_star = phys.LASER_PULSE_PERIOD / self.T_char
        # self.sigma0_star = phys.SIGMA0 / self.L_char
        
        # # Безразмерные границы области 
        # self.x_min_star = -1.0
        # self.x_max_star = 1.0
        # self.y_min_star = -1.0
        # self.y_max_star = 1.0
        # self.z_min_star = 0.0
        # self.z_max_star = 1.0
        # self.t_max_star = phys.T_MAX / self.T_char
        
        # # Масштабы для обратного преобразования (ДОБАВИТЬ)
        # self.x_scale = phys.X_MAX - phys.X_MIN
        # self.y_scale = phys.Y_MAX - phys.Y_MIN
        # self.z_scale = phys.Z_MAX - phys.Z_MIN
        
        # self._print_parameters()
    
    # def _print_parameters(self):
    #     """Вывод параметров нормировки"""
    #     print("=== ПАРАМЕТРЫ НОРМИРОВКИ ===")
    #     print(f"Пространственный масштаб: L_char = {self.L_char*1e6:.1f} мкм")
    #     print(f"Толщина образца: H_char = {self.H_char*1e6:.1f} мкм")
    #     print(f"Временной масштаб: T_char = {self.T_char*1e6:.1f} мкс")
    #     print(f"Температурный масштаб: T_scale = {self.T_scale:.2f} K")
    #     print(f"Безразмерный коэффициент диффузии: α* = {self.alpha_star:.3f}")
    #     print(f"Безразмерный коэффициент поглощения: μ* = {self.mu_star:.3f}")  # ← ДОБАВИТЬ
    #     print(f"Безразмерная амплитуда лазера: Q* = {self.laser_amplitude_star:.3f}")
    
    # def to_dimensionless(self, x=None, y=None, z=None, t=None, T=None, Q=None):
    #     """Преобразование в безразмерные величины [0,1]"""
    #     result = {}
    #     if x is not None: 
    #         result['x_star'] = (x - phys.X_MIN) / self.x_scale  # → [0,1]
    #     if y is not None: 
    #         result['y_star'] = (y - phys.Y_MIN) / self.y_scale  # → [0,1]
    #     if z is not None: 
    #         result['z_star'] = (z - phys.Z_MIN) / self.z_scale  # → [0,1]
    #     if t is not None: 
    #         result['t_star'] = t / self.T_char
    #     if T is not None: 
    #         result['T_star'] = T / self.T_scale
    #     if Q is not None: 
    #         result['Q_star'] = Q / self.Q_scale
    #     return result
    
    # def to_dimensional(self, x_star=None, y_star=None, z_star=None, t_star=None, T_star=None, Q_star=None):
    #     """Преобразование в размерные величины"""
    #     result = {}
    #     if x_star is not None: 
    #         result['x'] = x_star * self.x_scale + phys.X_MIN  # ← ИСПРАВИТЬ
    #     if y_star is not None: 
    #         result['y'] = y_star * self.y_scale + phys.Y_MIN  # ← ИСПРАВИТЬ
    #     if z_star is not None: 
    #         result['z'] = z_star * self.z_scale + phys.Z_MIN  # ← ИСПРАВИТЬ
    #     if t_star is not None: 
    #         result['t'] = t_star * self.T_char
    #     if T_star is not None: 
    #         result['T'] = T_star * self.T_scale
    #     if Q_star is not None: 
    #         result['Q'] = Q_star * self.Q_scale
    #     return result
    
    # def normalize_tensor(self, x_tensor, y_tensor, z_tensor, t_tensor, T_tensor=None, Q_tensor=None, mode='to_dimensionless'):
    #     """Нормировка тензоров PyTorch"""
    #     if mode == 'to_dimensionless':
    #         x_norm = (x_tensor - phys.X_MIN) / self.x_scale  # ← ИСПРАВИТЬ
    #         y_norm = (y_tensor - phys.Y_MIN) / self.y_scale
    #         z_norm = (z_tensor - phys.Z_MIN) / self.z_scale
    #         t_norm = t_tensor / self.T_char
    #         result = (x_norm, y_norm, z_norm, t_norm)
    #         if T_tensor is not None:
    #             result += (T_tensor / self.T_scale,)
    #         if Q_tensor is not None:
    #             result += (Q_tensor / self.Q_scale,)
    #     else:  # to_dimensional
    #         x_dim = x_tensor * self.x_scale + phys.X_MIN  # ← ИСПРАВИТЬ
    #         y_dim = y_tensor * self.y_scale + phys.Y_MIN
    #         z_dim = z_tensor * self.z_scale + phys.Z_MIN
    #         t_dim = t_tensor * self.T_char
    #         result = (x_dim, y_dim, z_dim, t_dim)
    #         if T_tensor is not None:
    #             result += (T_tensor * self.T_scale,)
    #         if Q_tensor is not None:
    #             result += (Q_tensor * self.Q_scale,)
    #     return result
    
    # def get_dimensionless_params(self):
    #     """Возвращает безразмерные параметры для PINN"""
    #     return {
    #         'alpha_star': self.alpha_star,
    #         'mu_star': self.mu_star,  # ← ДОБАВИТЬ
    #         'laser_amplitude_star': self.laser_amplitude_star,
    #         'laser_sigma_star': self.laser_sigma_star,
    #         'pulse_duration_star': self.pulse_duration_star,
    #         'pulse_period_star': self.pulse_period_star,
    #         'sigma0_star': self.sigma0_star,
    #         'x_min_star': self.x_min_star,  # теперь [0,1]
    #         'x_max_star': self.x_max_star,
    #         'y_min_star': self.y_min_star,
    #         'y_max_star': self.y_max_star,
    #         'z_min_star': self.z_min_star,
    #         'z_max_star': self.z_max_star,
    #         't_max_star': self.t_max_star
    #     }
    
    @staticmethod
    def normalize_vector(*data_args, get_coef: bool = False) -> tuple[torch.Tensor, ...]:
        results = []
        coef = []
        
        for data in data_args:
            if not isinstance(data, torch.Tensor):
                data_tensor = torch.tensor(data, dtype=torch.float32)
            else:
                data_tensor = data.clone().detach().float()
            
            if data_tensor.numel() == 0:
                results.append(data_tensor)
                continue
            
            min_val = torch.min(data_tensor)
            max_val = torch.max(data_tensor)
            
            # Если все элементы одинаковые, возвращаем тензор из нулей
            if max_val == min_val:
                results.append(torch.zeros_like(data_tensor))
                coef.append(torch.zeros_like(data_tensor))
            else:
                result_tensor = (data_tensor - min_val) / (max_val - min_val)
                results.append(result_tensor)
                coef.append(result_tensor / data_tensor)


        if get_coef:
            return tuple(zip(results, coef))
        else:    
            return tuple(results)
        

# (x_1, x_2), (y_1, y_2) = QuartzNormalizer.normalize_vector(np.array([1,2,3,4,5,6]), [3,56,7,3,2], get_coef=True)   
# print(x_1)
# print(x_2)     