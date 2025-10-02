# physical_params.py
"""
Физические параметры кварца JGS1 и параметры лазера
Все величины в СИ
"""
import numpy as np

# Физические параметры кварца JGS1
THERMAL_DIFFUSIVITY = 8.4e-7  # м²/с
THERMAL_CONDUCTIVITY = 1.4     # Вт/(м·К)
DENSITY = 2200                 # кг/м³
SPECIFIC_HEAT = 670           # Дж/(кг·К)

# Параметры лазера
LASER_PULSE_DURATION = 15e-6   # 15 мкс
LASER_PULSE_PERIOD = 110e-6    # 110 мкс
LASER_AMPLITUDE = 1e10         # амплитуда источника тепла (Вт/м³)
LASER_SIGMA = 10e-6            
LASER_WAVELENGTH = 10600e-9    # длина волны 10600 нм
POWER = 11,1                    # мощность 11,1 Вт
W0 = 10e-6                         # радиус пучка 10 мкм (1/e²)

#вариант для расчета I0
I0 = POWER / (np.pi * W0**2)


# Оптические параметры кварца JGS1
ABSORPTION_COEFFICIENT = 0.9   # коэффициент поглощения, 1/м
REFLECTIVITY = 0.04            # коэффициент отражения


# Геометрические параметры
X_MIN = -50e-6    # -50 мкм
X_MAX = 50e-6     # 50 мкм  
Y_MIN = -50e-6    # -50 мкм
Y_MAX = 50e-6     # 50 мкм
Z_MIN = 0         # 0 мкм
Z_MAX = 100e-6    # 100 мкм

MU: float = 1000.0

# Временные параметры
T_MAX = 500e-6    # общее время моделирования 500 мкс

# Параметры PINN
NUM_GAUSSIANS = 1
GAUSSIAN_SPACING = 0.5
SIGMA0 = 10e-6    # начальная ширина гауссиана