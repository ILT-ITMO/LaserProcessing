# =====================================================
# ФИЗИЧЕСКИЕ ПАРАМЕТРЫ (СИ)
# =====================================================

# Параметры лазера
LASER_WAVELENGTH = 10.6e-6      # Длина волны [м]
LASER_REP_RATE = 8000.0         # Частота повторения [Гц]  
LASER_PULSE_DURATION = 15e-6    # Длительность импульса FWHM [с]
LASER_AVG_POWER = 10.0          # Средняя мощность [Вт]
LASER_BEAM_RADIUS = 62e-6       # Радиус пучка [м]
LASER_SCAN_VELOCITY = 0.06      # Скорость сканирования [м/с]

# Параметры материала (Кварц JS1)
MATERIAL_DENSITY = 2200.0       # Плотность [кг/м^3]
MATERIAL_SPECIFIC_HEAT = 670.0  # Удельная теплоемкость [Дж/(кг·К)]
MATERIAL_CONDUCTIVITY = 1.4     # Теплопроводность [Вт/(м·К)]
MATERIAL_ABSORPTION = 5000.0    # Коэффициент поглощения [1/м]
MATERIAL_REFLECTIVITY = 0.25    # Коэффициент отражения

# Начальные условия
INITIAL_TEMPERATURE = 300.0     # Начальная температура [K]

# =====================================================
# ВЫЧИСЛЕННЫЕ ПАРАМЕТРЫ
# =====================================================
import math

# Лазерные параметры
LASER_PULSE_PERIOD = 1.0 / LASER_REP_RATE  # Период [с]
LASER_DUTY_CYCLE = LASER_REP_RATE * LASER_PULSE_DURATION  # Скважность
LASER_PEAK_POWER = LASER_AVG_POWER / LASER_DUTY_CYCLE  # Пиковая мощность [Вт]
LASER_PEAK_INTENSITY = (2 * LASER_PEAK_POWER) / (math.pi * LASER_BEAM_RADIUS**2)  # Пиковая интенсивность [Вт/м^2]

# Параметры гауссова импульса (FWHM -> sigma)
LASER_PULSE_SIGMA = LASER_PULSE_DURATION / (2 * math.sqrt(2 * math.log(2)))  # Стандартное отклонение [с]

# Теплофизические параметры
THERMAL_DIFFUSIVITY = MATERIAL_CONDUCTIVITY / (MATERIAL_DENSITY * MATERIAL_SPECIFIC_HEAT)  # Температуропроводность [м^2/с]

# =====================================================
# МАСШТАБИРОВАНИЕ ДЛЯ БЕЗРАЗМЕРНОЙ PINN
# =====================================================
# Характерные масштабы
CHARACTERISTIC_LENGTH = LASER_BEAM_RADIUS  # Характерная длина [м]
CHARACTERISTIC_TIME = CHARACTERISTIC_LENGTH**2 / THERMAL_DIFFUSIVITY  # Характерное время [с]
CHARACTERISTIC_TEMPERATURE = (1 - MATERIAL_REFLECTIVITY) * LASER_PEAK_INTENSITY * MATERIAL_ABSORPTION * CHARACTERISTIC_LENGTH**2 / MATERIAL_CONDUCTIVITY  # Характерная температура [K]

# Безразмерные параметры для PINN
NUM_GAUSSIANS = 1
GAUSSIAN_SPACING = 0.5
SIGMA0 = 0.1

# Безразмерные лазерные параметры
LASER_PULSE_DURATION_NORM = LASER_PULSE_DURATION / CHARACTERISTIC_TIME
LASER_PULSE_PERIOD_NORM = LASER_PULSE_PERIOD / CHARACTERISTIC_TIME
LASER_PULSE_SIGMA_NORM = LASER_PULSE_SIGMA / CHARACTERISTIC_TIME
LASER_SIGMA_NORM = 1.0  # Нормированный размер пучка

# ДОБАВЛЯЕМ НОВЫЙ ПАРАМЕТР:
LASER_AMPLITUDE = 1.0  # Безразмерная амплитуда источника тепла

print(f"Характерная длина: {CHARACTERISTIC_LENGTH*1e6:.2f} мкм")
print(f"Характерное время: {CHARACTERISTIC_TIME*1e3:.2f} мс") 
print(f"Характерная температура: {CHARACTERISTIC_TEMPERATURE:.2f} K")
print(f"Пиковая интенсивность: {LASER_PEAK_INTENSITY/1e6:.2f} МВт/м²")
print(f"Безразмерная амплитуда лазера: {LASER_AMPLITUDE}")
print(f"Длительность импульса (FWHM): {LASER_PULSE_DURATION*1e6:.1f} мкс")
print(f"Стандартное отклонение импульса: {LASER_PULSE_SIGMA*1e6:.1f} мкс")