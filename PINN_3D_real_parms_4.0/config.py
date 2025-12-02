import json
import math
import os

class LaserConfig:
    """Класс для хранения и управления конфигурацией лазерного нагрева"""
    
    def __init__(self, config_file=None):
        """
        Инициализация конфигурации
        
        Args:
            config_file: путь к JSON файлу с конфигурацией (опционально)
        """
        # Значения по умолчанию
        self.default_config = {
            # Физические параметры лазера
            "laser": {
                "wavelength": 10.6e-6,          # Длина волны [м]
                "rep_rate": 8000.0,             # Частота повторения [Гц]
                "pulse_duration": 15e-6,        # Длительность импульса FWHM [с]
                "avg_power": 10.0,              # Средняя мощность [Вт]
                "beam_radius": 62e-6,           # Радиус пучка [м]
                "scan_velocity": 0.06,          # Скорость сканирования [м/с]
                "mode": "pulsed",               # Режим: "pulsed" или "continuous"
                "continuous_power": 10.0,       # Мощность непрерывного лазера [Вт]
                "num_pulses": 8,                # Количество импульсов (для импульсного режима)
                "simulation_time": None         # Время моделирования [с] (None - вычисляется автоматически)
            },
            
            # Параметры материала
            "material": {
                "density": 2200.0,              # Плотность [кг/м^3]
                "specific_heat": 670.0,         # Удельная теплоемкость [Дж/(кг·К)]
                "conductivity": 1.4,            # Теплопроводность [Вт/(м·К)]
                "absorption": 5000.0,           # Коэффициент поглощения [1/м]
                "reflectivity": 0.25,           # Коэффициент отражения
                "initial_temperature": 300.0    # Начальная температура [K]
            },
            
            # Параметры PINN
            "pinn": {
                "num_gaussians": 1,
                "gaussian_spacing": 0.5,
                "sigma0": 0.1,
                "laser_amplitude": 1.0,
                "collocation_points": {"x": 20, "y": 20, "z": 20, "t": 20},
                "visualization_points": {"x": 30, "y": 30, "z": 30, "t": 20}
            },
            
            # Параметры обучения
            "training": {
                "num_epochs": 1000,
                "learning_rate": 1e-3,
                "device": "auto",  # "auto", "cpu", "cuda", "mps"
                "loss_weights": {"pde": 1.0, "ic": 1.0, "bc": 2.0}
            }
        }
        
        # Загружаем конфигурацию из файла если указан
        if config_file and os.path.exists(config_file):
            self.load_from_json(config_file)
        else:
            self.config = self.default_config.copy()
        
        # Вычисляем производные параметры
        self.calculate_derived_parameters()
    
    def load_from_json(self, filepath):
        """Загрузить конфигурацию из JSON файла"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                loaded_config = json.load(f)
            
            # Рекурсивно обновляем конфигурацию
            self.config = self.deep_update(self.default_config.copy(), loaded_config)
            print(f"Конфигурация загружена из {filepath}")
            
        except Exception as e:
            print(f"Ошибка загрузки конфигурации из {filepath}: {e}")
            print("Используются значения по умолчанию")
            self.config = self.default_config.copy()
    
    def deep_update(self, base_dict, update_dict):
        """Рекурсивно обновляет словарь"""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                base_dict[key] = self.deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
        return base_dict
    
    def save_to_json(self, filepath):
        """Сохранить конфигурацию в JSON файл"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False, default=str)
            print(f"Конфигурация сохранена в {filepath}")
        except Exception as e:
            print(f"Ошибка сохранения конфигурации: {e}")
    
    def calculate_derived_parameters(self):
        """Вычислить производные параметры"""
        laser = self.config["laser"]
        material = self.config["material"]
        
        # Устанавливаем режим
        self.LASER_MODE = laser["mode"]
        
        # Выбор режима
        if self.LASER_MODE == "continuous":
            # Для непрерывного режима используем среднюю мощность как постоянную
            self.LASER_PEAK_POWER = laser["continuous_power"]
            self.LASER_PULSE_DURATION = 1e-6  # малая длительность для вычислений
            self.LASER_REP_RATE = 1.0  # частота 1 Гц для упрощения
            self.LASER_AVG_POWER = laser["continuous_power"]
        else:
            # Для импульсного режима используем вычисленные параметры
            self.LASER_AVG_POWER = laser["avg_power"]
            self.LASER_REP_RATE = laser["rep_rate"]
            self.LASER_PULSE_DURATION = laser["pulse_duration"]
            self.LASER_PEAK_POWER = laser["avg_power"] / (laser["rep_rate"] * laser["pulse_duration"])
        
        # Общие вычисления
        self.LASER_PULSE_PERIOD = 1.0 / self.LASER_REP_RATE
        self.LASER_DUTY_CYCLE = self.LASER_REP_RATE * self.LASER_PULSE_DURATION
        self.LASER_PEAK_INTENSITY = (2 * self.LASER_PEAK_POWER) / (math.pi * laser["beam_radius"]**2)
        
        # Параметры гауссова импульса
        self.LASER_PULSE_SIGMA = self.LASER_PULSE_DURATION / (2 * math.sqrt(2 * math.log(2)))
        
        # Теплофизические параметры
        self.THERMAL_DIFFUSIVITY = material["conductivity"] / (material["density"] * material["specific_heat"])
        
        # Характерные масштабы
        self.CHARACTERISTIC_LENGTH = laser["beam_radius"]
        self.CHARACTERISTIC_TIME = self.CHARACTERISTIC_LENGTH**2 / self.THERMAL_DIFFUSIVITY
        self.CHARACTERISTIC_TEMPERATURE = (
            (1 - material["reflectivity"]) * 
            self.LASER_PEAK_INTENSITY * 
            material["absorption"] * 
            self.CHARACTERISTIC_LENGTH**2 / 
            material["conductivity"]
        )
        
        # Безразмерные параметры
        self.LASER_AMPLITUDE = self.config["pinn"]["laser_amplitude"]
        self.LASER_PULSE_DURATION_NORM = self.LASER_PULSE_DURATION / self.CHARACTERISTIC_TIME
        self.LASER_PULSE_PERIOD_NORM = self.LASER_PULSE_PERIOD / self.CHARACTERISTIC_TIME
        self.LASER_PULSE_SIGMA_NORM = self.LASER_PULSE_SIGMA / self.CHARACTERISTIC_TIME
        self.LASER_SIGMA_NORM = 1.0
        
        # Время моделирования
        if laser["simulation_time"] is not None:
            # Если время задано явно
            self.SIMULATION_TIME_PHYSICAL = laser["simulation_time"]
        else:
            # Автоматический расчет
            if self.LASER_MODE == "pulsed":
                self.SIMULATION_TIME_PHYSICAL = laser["num_pulses"] * self.LASER_PULSE_PERIOD
                self.NUM_PULSES = laser["num_pulses"]
            else:
                # Для непрерывного режима используем характерное время
                self.SIMULATION_TIME_PHYSICAL = self.CHARACTERISTIC_TIME
                self.NUM_PULSES = 1
        
        self.SIMULATION_TIME_NORM = self.SIMULATION_TIME_PHYSICAL / self.CHARACTERISTIC_TIME
        
        # Материал
        self.MATERIAL_DENSITY = material["density"]
        self.MATERIAL_SPECIFIC_HEAT = material["specific_heat"]
        self.MATERIAL_CONDUCTIVITY = material["conductivity"]
        self.MATERIAL_ABSORPTION = material["absorption"]
        self.MATERIAL_REFLECTIVITY = material["reflectivity"]
        self.INITIAL_TEMPERATURE = material["initial_temperature"]
        
        # PINN параметры
        self.NUM_GAUSSIANS = self.config["pinn"]["num_gaussians"]
        self.GAUSSIAN_SPACING = self.config["pinn"]["gaussian_spacing"]
        self.SIGMA0 = self.config["pinn"]["sigma0"]
        
        # Другие параметры для обратной совместимости
        self.LASER_WAVELENGTH = laser["wavelength"]
        self.LASER_BEAM_RADIUS = laser["beam_radius"]
        self.LASER_SCAN_VELOCITY = laser["scan_velocity"]
        self.LASER_CONTINUOUS_POWER = laser["continuous_power"]
    
    def print_summary(self):
        """Вывести сводку конфигурации"""
        print("=" * 60)
        print("КОНФИГУРАЦИЯ МОДЕЛИ ЛАЗЕРНОГО НАГРЕВА")
        print("=" * 60)
        print(f"Режим лазера: {self.LASER_MODE.upper()}")
        print(f"Материал: кварц JS1")
        print(f"Начальная температура: {self.INITIAL_TEMPERATURE} K")
        print()
        
        print("Характерные масштабы:")
        print(f"  Длина: {self.CHARACTERISTIC_LENGTH*1e6:.2f} мкм")
        print(f"  Время: {self.CHARACTERISTIC_TIME*1e3:.2f} мс")
        print(f"  Температура: {self.CHARACTERISTIC_TEMPERATURE:.1f} K")
        print()
        
        if self.LASER_MODE == "pulsed":
            print("Импульсный режим:")
            print(f"  Средняя мощность: {self.LASER_AVG_POWER} Вт")
            print(f"  Частота: {self.LASER_REP_RATE:.0f} Гц")
            print(f"  Длительность импульса: {self.LASER_PULSE_DURATION*1e6:.1f} мкс")
            print(f"  Количество импульсов: {self.NUM_PULSES}")
            print(f"  Пиковая мощность: {self.LASER_PEAK_POWER:.1f} Вт")
            print(f"  Пиковая интенсивность: {self.LASER_PEAK_INTENSITY/1e6:.1f} МВт/м²")
        else:
            print("Непрерывный режим:")
            print(f"  Мощность: {self.LASER_CONTINUOUS_POWER} Вт")
            print(f"  Интенсивность: {self.LASER_PEAK_INTENSITY/1e6:.1f} МВт/м²")
        
        print()
        print("Время моделирования:")
        print(f"  Физическое: {self.SIMULATION_TIME_PHYSICAL*1e6:.1f} мкс")
        print(f"  Безразмерное: {self.SIMULATION_TIME_NORM:.3f}")
        print()
        
        print("Параметры PINN:")
        print(f"  Безразмерная амплитуда лазера: {self.LASER_AMPLITUDE}")
        print(f"  Точки коллокации: {self.config['pinn']['collocation_points']}")
        print(f"  Точки визуализации: {self.config['pinn']['visualization_points']}")
        print("=" * 60)

# Создаем глобальный объект конфигурации
config_manager = LaserConfig()

# Для обратной совместимости экспортируем все атрибуты как глобальные переменные
def export_globals():
    """Экспортировать все параметры как глобальные переменные"""
    import sys
    module = sys.modules[__name__]
    
    for attr_name in dir(config_manager):
        if not attr_name.startswith('_') and not callable(getattr(config_manager, attr_name)):
            attr_value = getattr(config_manager, attr_name)
            if not isinstance(attr_value, dict) and not isinstance(attr_value, list):
                setattr(module, attr_name, attr_value)
    
    # Экспортируем сам объект для доступа к полной конфигурации
    setattr(module, 'CONFIG', config_manager)

export_globals()

# Автоматический вывод сводки при импорте
if __name__ != "__main__":
    config_manager.print_summary()