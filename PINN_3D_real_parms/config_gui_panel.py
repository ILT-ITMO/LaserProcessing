# config_gui_panel.py
import panel as pn
import param
import json
import numpy as np
from pathlib import Path
import config

# Инициализация Panel
pn.extension(design='material', loading_indicator=True)

# ============================================================================
# ОСНОВНОЙ КЛАСС ПРИЛОЖЕНИЯ
# ============================================================================

class LaserConfigApp(param.Parameterized):
    """
    Класс приложения для настройки конфигурации лазерного нагрева
    """

    
    # === ПАРАМЕТРЫ ЛАЗЕРА ===
    laser_mode = param.Selector(
        objects=['pulsed', 'continuous'],
        default='continuous',
        label="📡 Режим лазера"
    )
    
    # Основные параметры
    laser_wavelength = param.Number(
        10.6e-6,
        bounds=(0.1e-6, 100e-6),
        label="📏 Длина волны (м)"
    )
    
    laser_beam_radius = param.Number(
        62e-6,
        bounds=(1e-6, 500e-6),
        label="🎯 Радиус пучка (м)"
    )
    
    laser_scan_velocity = param.Number(
        0.06,
        bounds=(0.0, 10.0),
        label="⚡ Скорость сканирования (м/с)"
    )
    
    # Параметры импульсного режима
    laser_rep_rate = param.Number(
        8000.0,
        bounds=(1.0, 100000.0),
        label="⏱️ Частота повторения (Гц)"
    )
    
    laser_pulse_duration = param.Number(
        15e-6,
        bounds=(1e-9, 100e-6),
        label="⌛ Длительность импульса (с)"
    )
    
    laser_avg_power = param.Number(
        10.0,
        bounds=(0.1, 1000.0),
        label="⚡ Средняя мощность (Вт)"
    )
    
    num_pulses = param.Integer(
        8,
        bounds=(1, 100),
        label="🔢 Количество импульсов"
    )
    
    # Параметры непрерывного режима
    laser_continuous_power = param.Number(
        5.0,
        bounds=(0.1, 1000.0),
        label="💡 Мощность непрерывного лазера (Вт)"
    )
    
    simulation_time = param.Number(
        2e-3,
        bounds=(1e-6, 10.0),
        label="⏰ Время моделирования (с)"
    )
    
    # === ПАРАМЕТРЫ МАТЕРИАЛА ===
    material_density = param.Number(
        2200.0,
        bounds=(100.0, 10000.0),
        label="⚖️ Плотность (кг/м³)"
    )
    
    material_specific_heat = param.Number(
        670.0,
        bounds=(100.0, 5000.0),
        label="🔥 Удельная теплоемкость (Дж/(кг·К))"
    )
    
    material_conductivity = param.Number(
        1.4,
        bounds=(0.1, 500.0),
        label="🌡️ Теплопроводность (Вт/(м·К))"
    )
    
    material_absorption = param.Number(
        5000.0,
        bounds=(1.0, 100000.0),
        label="🎯 Коэффициент поглощения (1/м)"
    )
    
    material_reflectivity = param.Number(
        0.25,
        bounds=(0.0, 1.0),
        label="✨ Коэффициент отражения"
    )
    
    initial_temperature = param.Number(
        300.0,
        bounds=(0.0, 5000.0),
        label="🌡️ Начальная температура (K)"
    )
    
    # === ПАРАМЕТРЫ PINN ===
    laser_amplitude = param.Number(
        1.0,
        bounds=(0.1, 10.0),
        label="📊 Безразмерная амплитуда лазера"
    )
    
    col_x = param.Integer(20, bounds=(5, 100), label="📐 Коллокация X")
    col_y = param.Integer(20, bounds=(5, 100), label="📐 Коллокация Y")
    col_z = param.Integer(20, bounds=(5, 100), label="📐 Коллокация Z")
    col_t = param.Integer(20, bounds=(5, 100), label="⏱️ Коллокация T")
    
    vis_x = param.Integer(30, bounds=(5, 100), label="👁️ Визуализация X")
    vis_y = param.Integer(30, bounds=(5, 100), label="👁️ Визуализация Y")
    vis_z = param.Integer(30, bounds=(5, 100), label="👁️ Визуализация Z")
    vis_t = param.Integer(20, bounds=(5, 100), label="⏱️ Визуализация T")
    
    # === ПАРАМЕТРЫ ОБУЧЕНИЯ ===
    num_epochs = param.Integer(
        1000,
        bounds=(100, 10000),
        label="🎯 Количество эпох"
    )
    
    learning_rate = param.Number(
        1e-3,
        bounds=(1e-5, 1e-1),
        label="📈 Learning rate"
    )
    
    loss_pde = param.Number(
        1.0,
        bounds=(0.1, 10.0),
        label="📐 PDE вес"
    )
    
    loss_ic = param.Number(
        1.0,
        bounds=(0.1, 10.0),
        label="🎯 IC вес"
    )
    
    loss_bc = param.Number(
        2.0,
        bounds=(0.1, 10.0),
        label="🔲 BC вес"
    )
    
    # === СВОЙСТВА ВЫЧИСЛЕНИЙ ===
    calculated_char_length = param.String("0.0 мкм", label="Характерная длина")
    calculated_char_time = param.String("0.0 мс", label="Характерное время")
    calculated_char_temp = param.String("0.0 K", label="Характерная температура")
    calculated_peak_intensity = param.String("0.0 МВт/м²", label="Пиковая интенсивность")
    
    # === ВЫХОДНЫЕ ДАННЫЕ ===
    json_output = param.String("", label="JSON конфигурация")
    status_message = param.String("Готово", label="Статус")
    
    def __init__(self, **params):
        """
        Initializes the LaserConfigApp object.
        
        This constructor sets up the application by initializing the base class, 
        loading the configuration, creating the user interface elements, and 
        determining the starting values for calculations. This ensures the application 
        is ready to process laser-related data and simulations.
        
        Args:
            self: The LaserConfigApp instance.
            **params: Keyword arguments passed to the superclass constructor.
        
        Initializes the following class fields:
            config: Stores the application's configuration settings.
            widgets: Stores the GUI widgets for user interaction.
            initial_parameters: Stores the initial values used in calculations.
        
        Returns:
            None
        """
        super().__init__(**params)
        self.load_current_config()
        self.create_widgets()
        self.calculate_initial_parameters()
    
    def load_current_config(self):
        """
        Loads the current configuration parameters for the laser processing simulation.
        
        This method retrieves and sets the simulation parameters from the configuration file, 
        including laser properties, material properties, and training settings. 
        These parameters are essential for defining the simulation environment and 
        controlling the behavior of the laser-material interaction model.
        
        Args:
            self: The LaserConfigApp instance.
        
        Returns:
            None
        """
        try:
            self.laser_mode = config.LASER_MODE
            self.laser_wavelength = config.LASER_WAVELENGTH
            self.laser_beam_radius = config.LASER_BEAM_RADIUS
            self.laser_scan_velocity = config.LASER_SCAN_VELOCITY
            
            self.laser_rep_rate = config.LASER_REP_RATE
            self.laser_pulse_duration = config.LASER_PULSE_DURATION
            self.laser_avg_power = config.LASER_AVG_POWER
            self.num_pulses = config.NUM_PULSES
            
            self.laser_continuous_power = config.LASER_CONTINUOUS_POWER
            self.simulation_time = config.SIMULATION_TIME_PHYSICAL
            
            self.material_density = config.MATERIAL_DENSITY
            self.material_specific_heat = config.MATERIAL_SPECIFIC_HEAT
            self.material_conductivity = config.MATERIAL_CONDUCTIVITY
            self.material_absorption = config.MATERIAL_ABSORPTION
            self.material_reflectivity = config.MATERIAL_REFLECTIVITY
            self.initial_temperature = config.INITIAL_TEMPERATURE
            
            self.laser_amplitude = config.LASER_AMPLITUDE
            
            coll_points = config.CONFIG.config["pinn"]["collocation_points"]
            self.col_x = coll_points["x"]
            self.col_y = coll_points["y"]
            self.col_z = coll_points["z"]
            self.col_t = coll_points["t"]
            
            vis_points = config.CONFIG.config["pinn"]["visualization_points"]
            self.vis_x = vis_points["x"]
            self.vis_y = vis_points["y"]
            self.vis_z = vis_points["z"]
            self.vis_t = vis_points["t"]
            
            train_config = config.CONFIG.config["training"]
            self.num_epochs = train_config["num_epochs"]
            self.learning_rate = train_config["learning_rate"]
            
            loss_weights = train_config["loss_weights"]
            self.loss_pde = loss_weights["pde"]
            self.loss_ic = loss_weights["ic"]
            self.loss_bc = loss_weights["bc"]
            
        except Exception as e:
            print(f"Ошибка загрузки конфигурации: {e}")
            self.update_status(f"Ошибка загрузки конфигурации: {str(e)}", "danger")
    
    # ============================================================================
    # СОЗДАНИЕ ВИДЖЕТОВ
    # ============================================================================
    
    def create_widgets(self):
        """
        Creates and arranges the user interface widgets for configuring laser processing parameters.
        
        This method constructs the main layout of the application, including a title pane, tabbed sections for different configuration aspects (laser, material, PINN settings, training parameters, calculations, and controls), and a status bar for displaying messages.  These widgets are organized to allow users to define and adjust the simulation setup.
        
        Args:
            self:  The LaserConfigApp instance.
        
        Returns:
            None: This method modifies the object's attributes (self.title_pane, self.tabs, self.status_bar, self.layout) directly and doesn't return a value.
        """
        
        # Заголовок приложения
        self.title_pane = pn.pane.HTML("""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 40px;
            border-radius: 10px;
            color: white;
            margin-bottom: 20px;
            text-align: center;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        ">
            <h1 style="margin: 0; font-size: 2.8em; font-weight: 700;">⚙️ Конфигуратор моделирования</h1>
            <p style="margin: 10px 0 0 0; font-size: 1.2em; opacity: 0.9;">
                Настройка параметров лазерного нагрева для PINN
            </p>
        </div>
        """)
        
        # Создаем вкладки
        self.tabs = pn.Tabs(
            ('⚡ Лазер', self.create_laser_tab()),
            ('🧱 Материал', self.create_material_tab()),
            ('🧮 PINN', self.create_pinn_tab()),
            ('🎓 Обучение', self.create_training_tab()),
            ('📊 Расчеты', self.create_calculations_tab()),
            ('🎯 Управление', self.create_controls_tab()),
            tabs_location='above',
            sizing_mode='stretch_width'
        )
        
        # Статус бар
        self.status_bar = pn.pane.Alert(
            self.status_message,
            alert_type="info",
            margin=(10, 0, 0, 0)
        )
        
        # Основной лейаут
        self.layout = pn.Column(
            self.title_pane,
            self.tabs,
            self.status_bar,
            sizing_mode='stretch_width',
            margin=(0, 20)
        )
    
    def create_laser_tab(self):
        """
        Создает вкладку настроек лазера, позволяющую пользователю конфигурировать параметры лазерного излучения, такие как режим работы (импульсный или непрерывный), длина волны, мощность и другие характеристики. Вкладка предоставляет интерактивные элементы управления для изменения параметров и отображает подсказки для типичных значений.
        """
        
        # Виджет выбора режима
        mode_selector = pn.widgets.RadioButtonGroup(
            name='Режим лазера',
            options=['Импульсный', 'Непрерывный'],
            value='Импульсный' if self.laser_mode == 'pulsed' else 'Непрерывный',
            button_type='success',
            margin=(0, 0, 20, 0)
        )
        
        # Привязываем к параметру
        def update_mode(event):
            self.laser_mode = 'pulsed' if event.new == 'Импульсный' else 'continuous'
            self.update_status(f"Режим изменен на: {event.new}")
        mode_selector.param.watch(update_mode, 'value')
        
        # Основные параметры в карточке
        basic_card = pn.Card(
            pn.Row(
                pn.Column(
                    pn.widgets.FloatInput.from_param(self.param.laser_wavelength),
                    pn.widgets.FloatInput.from_param(self.param.laser_beam_radius),
                    pn.widgets.FloatInput.from_param(self.param.laser_scan_velocity),
                ),
                pn.Spacer(width=20),
                pn.pane.HTML("""
                <div style="color: #666; font-size: 0.9em;">
                    <h4>💡 Подсказки:</h4>
                    <ul>
                        <li>Длина волны CO₂ лазера: 10.6 мкм</li>
                        <li>Типичный радиус пучка: 50-100 мкм</li>
                        <li>Скорость сканирования: 0.01-0.1 м/с</li>
                    </ul>
                </div>
                """)
            ),
            title="📡 Основные параметры лазера",
            collapsed=False,
            margin=(10, 0)
        )
        
        # Параметры импульсного режима
        pulsed_card = pn.Card(
            pn.Column(
                pn.widgets.FloatInput.from_param(self.param.laser_rep_rate),
                pn.widgets.FloatInput.from_param(self.param.laser_pulse_duration),
                pn.widgets.FloatInput.from_param(self.param.laser_avg_power),
                pn.widgets.IntInput.from_param(self.param.num_pulses),
            ),
            title="📈 Параметры импульсного режима",
            collapsed=(self.laser_mode != 'pulsed'),
            margin=(10, 0)
        )
        
        # Параметры непрерывного режима
        continuous_card = pn.Card(
            pn.Column(
                pn.widgets.FloatInput.from_param(self.param.laser_continuous_power),
                pn.widgets.FloatInput.from_param(self.param.simulation_time),
            ),
            title="🔆 Параметры непрерывного режима",
            collapsed=(self.laser_mode != 'continuous'),
            margin=(10, 0)
        )
        
        # Обновляем видимость карточек при изменении режима
        def update_cards_visibility(event):
            if event.new == 'pulsed':
                pulsed_card.collapsed = False
                continuous_card.collapsed = True
            else:
                pulsed_card.collapsed = True
                continuous_card.collapsed = False
        
        self.param.watch(update_cards_visibility, 'laser_mode')
        
        return pn.Column(
            pn.pane.HTML("<h3>⚡ Настройки лазерного излучения</h3>"),
            mode_selector,
            basic_card,
            pulsed_card,
            continuous_card,
            sizing_mode='stretch_width'
        )
    
    def create_material_tab(self):
        """
        Creates a tab for configuring material properties.
        
        This tab provides a user interface to adjust key material parameters used in laser-material interaction simulations. These parameters define how the material responds to laser heating, influencing the accuracy of thermal modeling.
        
        Args:
            self:  The LaserConfigApp instance.
        
        Returns:
            A Panel layout representing the material configuration tab. This layout includes input widgets for density, specific heat, thermal conductivity, absorption, reflectivity, and initial temperature, along with reference information about typical material properties.
        """
        
        # Сетка параметров материала
        material_grid = pn.GridSpec(ncols=2, sizing_mode='stretch_width')
        
        # Первая колонка
        material_grid[0:2, 0] = pn.Column(
            pn.widgets.FloatInput.from_param(self.param.material_density),
            pn.widgets.FloatInput.from_param(self.param.material_specific_heat),
            pn.widgets.FloatInput.from_param(self.param.material_conductivity),
            margin=(0, 10)
        )
        
        # Вторая колонка
        material_grid[0:2, 1] = pn.Column(
            pn.widgets.FloatInput.from_param(self.param.material_absorption),
            pn.widgets.FloatSlider.from_param(self.param.material_reflectivity),
            pn.widgets.FloatInput.from_param(self.param.initial_temperature),
            margin=(0, 10)
        )
        
        # Карточка с информацией о материале
        info_card = pn.Card(
            pn.pane.HTML("""
            <div style="color: #666; font-size: 0.95em; line-height: 1.6;">
                <h4>🧱 Кварц JS1 (типичные параметры):</h4>
                <ul>
                    <li><strong>Плотность:</strong> 2200-2500 кг/м³</li>
                    <li><strong>Теплопроводность:</strong> 1.3-1.5 Вт/(м·К)</li>
                    <li><strong>Теплоемкость:</strong> 670-750 Дж/(кг·К)</li>
                    <li><strong>Поглощение (10.6 мкм):</strong> 4000-6000 1/м</li>
                    <li><strong>Отражение:</strong> 0.2-0.3</li>
                </ul>
            </div>
            """),
            title="📋 Справочная информация",
            collapsed=False,
            margin=(20, 0, 0, 0)
        )
        
        return pn.Column(
            pn.pane.HTML("<h3>🧱 Настройки материала</h3>"),
            pn.Card(material_grid, title="📊 Параметры материала", collapsed=False),
            info_card,
            sizing_mode='stretch_width'
        )
    
    def create_pinn_tab(self):
        """
        Creates a configuration tab for PINN settings.
        
        This tab allows users to adjust parameters related to the Physics-Informed Neural Network (PINN) model, including laser amplitude, collocation grid resolution, and visualization grid resolution. These settings are crucial for accurately simulating and analyzing laser-induced heat transfer.
        
        Args:
            self:  An instance of the LaserConfigApp class.
        
        Returns:
            pn.Column: A Panel layout containing the PINN configuration widgets, 
                       organized into cards for basic parameters, collocation grid, 
                       and visualization grid.
        """
        
        # Карточка с основными параметрами PINN
        basic_card = pn.Card(
            pn.Column(
                pn.widgets.FloatSlider.from_param(self.param.laser_amplitude),
                margin=(10, 0)
            ),
            title="📈 Основные параметры PINN",
            collapsed=False,
            margin=(10, 0)
        )
        
        # Карточка сетки коллокации
        collocation_card = pn.Card(
            pn.GridBox(
                pn.widgets.IntSlider.from_param(self.param.col_x),
                pn.widgets.IntSlider.from_param(self.param.col_y),
                pn.widgets.IntSlider.from_param(self.param.col_z),
                pn.widgets.IntSlider.from_param(self.param.col_t),
                ncols=2,
                align='start'
            ),
            title="📐 Сетка коллокационных точек",
            collapsed=False,
            margin=(10, 0)
        )
        
        # Карточка сетки визуализации
        visualization_card = pn.Card(
            pn.GridBox(
                pn.widgets.IntSlider.from_param(self.param.vis_x),
                pn.widgets.IntSlider.from_param(self.param.vis_y),
                pn.widgets.IntSlider.from_param(self.param.vis_z),
                pn.widgets.IntSlider.from_param(self.param.vis_t),
                ncols=2,
                align='start'
            ),
            title="👁️ Сетка точек визуализации",
            collapsed=False,
            margin=(10, 0)
        )
        
        return pn.Column(
            pn.pane.HTML("<h3>🧮 Настройки PINN</h3>"),
            basic_card,
            collocation_card,
            visualization_card,
            sizing_mode='stretch_width'
        )
    
    def create_training_tab(self):
        """
        Creates a training settings tab with adjustable parameters for the simulation.
        
        The tab allows users to configure core training settings like the number of epochs and learning rate, as well as adjust the weights assigned to different loss functions representing physical constraints (PDE, initial conditions, boundary conditions). This configuration is crucial for effectively training models to accurately simulate laser-material interactions.
        
        Args:
            self:  The LaserConfigApp instance.
        
        Returns:
            pn.Column: A Panel layout representing the training settings tab, 
                       containing widgets for parameter adjustment and informative tooltips.
        """
        
        # Основные параметры обучения
        basic_params = pn.Row(
            pn.Column(
                pn.widgets.IntSlider.from_param(self.param.num_epochs),
                pn.widgets.FloatSlider.from_param(
                    self.param.learning_rate,
                    format="%.1e"
                ),
                width=300
            ),
            pn.Spacer(width=20),
            pn.pane.HTML("""
            <div style="background: #f8f9fa; padding: 20px; border-radius: 8px; border-left: 4px solid #4285f4;">
                <h4>💡 Рекомендации:</h4>
                <ul style="color: #666;">
                    <li>Для простых задач: 1000-2000 эпох</li>
                    <li>Для сложных задач: 5000-10000 эпох</li>
                    <li>Learning rate: 1e-4 - 1e-3</li>
                    <li>Используйте callback для ранней остановки</li>
                </ul>
            </div>
            """)
        )
        
        # Карточка с весами loss функций
        loss_cards = pn.GridBox(
            self.create_loss_card("📐 PDE", "#4285f4", self.param.loss_pde, 
                                 "Уравнение теплопроводности"),
            self.create_loss_card("🎯 IC", "#EA4335", self.param.loss_ic, 
                                 "Начальные условия"),
            self.create_loss_card("🔲 BC", "#34A853", self.param.loss_bc, 
                                 "Граничные условия"),
            ncols=3,
            align='start'
        )
        
        return pn.Column(
            pn.pane.HTML("<h3>🎓 Настройки обучения</h3>"),
            pn.Card(basic_params, title="⚙️ Основные параметры", collapsed=False),
            pn.Card(loss_cards, title="⚖️ Веса функций потерь", collapsed=False),
            sizing_mode='stretch_width'
        )
    
    def create_loss_card(self, title, color, parameter, description):
        """
        Создает карточку для веса loss функции с использованием интерактивного слайдера.
        
        Args:
            title (str): Заголовок карточки, отображаемый в верхней части.
            color (str): Цвет карточки и элементов управления.
            parameter (pn.param.Parameterized): Параметр, для которого создается слайдер.
            description (str): Описание параметра, отображаемое под заголовком.
        
        Returns:
            pn.Card: Карточка Panel с заголовком, описанием и слайдером для настройки веса loss функции.
        
        Метод создает визуальный элемент управления для настройки веса функции потерь, позволяя пользователю интерактивно изменять значение параметра с помощью слайдера. Это необходимо для точной настройки процесса обучения модели и достижения оптимальных результатов.
        """
        return pn.Card(
            pn.Column(
                pn.pane.HTML(f"""
                <div style="text-align: center; margin-bottom: 15px;">
                    <div style="color: {color}; font-weight: bold; font-size: 1.2em;">
                        {title}
                    </div>
                    <div style="color: #666; font-size: 0.9em; margin-top: 5px;">
                        {description}
                    </div>
                </div>
                """),
                pn.widgets.FloatSlider.from_param(parameter),
                align='center'
            ),
            styles={'background': f'{color}10', 'border': f'2px solid {color}'},
            margin=(5, 5),
            sizing_mode='stretch_height'
        )
    
    def create_calculations_tab(self):
        """
        Создает вкладку с результатами расчетов ключевых параметров, характеризующих взаимодействие лазера с материалом.
        
        Args:
            self: Экземпляр класса LaserConfigApp, содержащий данные и методы для конфигурации и управления приложением.
        
        Returns:
            pn.Column: Панель, содержащая результаты расчетов (характерная длина, время, температура, пиковая интенсивность), кнопку для пересчета параметров и информационную панель с пояснениями к расчетам.
        """
        
        # Карточки с результатами расчетов
        results_grid = pn.GridBox(
            self.create_result_card("📏 Характерная длина", self.calculated_char_length, 
                                   "#667eea", "мкм"),
            self.create_result_card("⏱️ Характерное время", self.calculated_char_time, 
                                   "#764ba2", "мс"),
            self.create_result_card("🌡️ Характерная температура", self.calculated_char_temp, 
                                   "#ff6b6b", "K"),
            self.create_result_card("⚡ Пиковая интенсивность", self.calculated_peak_intensity, 
                                   "#4ecdc4", "МВт/м²"),
            ncols=2,
            align='start'
        )
        
        # Кнопка пересчета
        calc_button = pn.widgets.Button(
            name="🧮 Пересчитать параметры",
            button_type="primary",
            width=200,
            margin=(20, 0, 0, 0)
        )
        calc_button.on_click(self.calculate_parameters)
        
        # Информационная панель
        info_pane = pn.pane.HTML("""
        <div style="background: #f8f9fa; padding: 20px; border-radius: 8px; margin-top: 20px;">
            <h4>📊 Пояснения к расчетам:</h4>
            <ul style="color: #666;">
                <li><strong>Характерная длина:</strong> радиус лазерного пучка</li>
                <li><strong>Характерное время:</strong> время тепловой диффузии через характерную длину</li>
                <li><strong>Характерная температура:</strong> максимальный перегрев от лазерного воздействия</li>
                <li><strong>Пиковая интенсивность:</strong> максимальная плотность мощности лазера</li>
            </ul>
        </div>
        """)
        
        return pn.Column(
            pn.pane.HTML("<h3>📊 Расчетные параметры</h3>"),
            pn.Card(results_grid, title="🔍 Результаты расчетов", collapsed=False),
            pn.Row(calc_button, align='center'),
            info_pane,
            sizing_mode='stretch_width'
        )
    
    def create_result_card(self, title, value_param, color, unit):
        """
        Создает карточку для отображения результата измерений или вычислений.
        
        Args:
            title (str): Заголовок карточки, описывающий отображаемый параметр.
            value_param (str): Значение параметра для отображения.
            color (str): Цвет карточки, используемый для визуального выделения.
            unit (str): Единица измерения параметра.
        
        Returns:
            pn.Card: Объект карточки Panel, готовый к отображению.
        """
        return pn.Card(
            pn.Column(
                pn.pane.HTML(f"""
                <div style="text-align: center; padding: 10px;">
                    <div style="color: {color}; font-weight: bold; font-size: 1.1em; margin-bottom: 10px;">
                        {title}
                    </div>
                    <div style="font-size: 1.8em; font-weight: 700; color: #2d3748;">
                        {value_param}
                    </div>
                    <div style="color: #718096; margin-top: 5px;">
                        {unit}
                    </div>
                </div>
                """),
                align='center'
            ),
            styles={'background': f'{color}10', 'border': f'2px solid {color}'},
            margin=(5, 5),
            sizing_mode='stretch_height'
        )
    
    def create_controls_tab(self):
        """
        Создает вкладку управления, содержащую кнопки для выполнения различных действий с конфигурацией лазера, таких как расчет параметров, генерация JSON, сохранение в файл и применение изменений. Также включает поле для отображения JSON конфигурации и панель быстрых действий для выбора предустановленных шаблонов.
        
        Args:
            None
        
        Returns:
            pn.Column: Панель с элементами управления конфигурацией лазера.
        """
        
        # Кнопки действий
        buttons_row = pn.Row(
            pn.widgets.Button(
                name="🧮 Рассчитать все",
                button_type="warning",
                width=150,
                margin=(5, 10)
            ),
            pn.widgets.Button(
                name="💾 Генерировать JSON",
                button_type="success",
                width=150,
                margin=(5, 10)
            ),
            pn.widgets.Button(
                name="📁 Сохранить в файл",
                button_type="primary",
                width=150,
                margin=(5, 10)
            ),
            pn.widgets.Button(
                name="🔄 Сбросить",
                button_type="light",
                width=150,
                margin=(5, 10)
            ),
            pn.widgets.Button(
                name="⚡ Применить",
                button_type="success",
                width=150,
                margin=(5, 10)
            ),
            align='center'
        )
        
        # Привязываем обработчики
        buttons_row[0].on_click(self.calculate_parameters)
        buttons_row[1].on_click(self.generate_json)
        buttons_row[2].on_click(self.save_to_file)
        buttons_row[3].on_click(self.reset_to_defaults)
        buttons_row[4].on_click(self.apply_config)
        
        # Поле вывода JSON
        json_output = pn.widgets.TextAreaInput(
            name='JSON конфигурация',
            value='',
            height=300,
            sizing_mode='stretch_width'
        )
        
        # Привязываем к параметру
        def update_json_output(event):
            json_output.value = event.new
        
        self.param.watch(update_json_output, 'json_output')
        
        # Панель быстрых действий
        quick_actions = pn.Card(
            pn.Column(
                pn.Row(
                    pn.widgets.Button(
                        name="🔄 Импульсный шаблон",
                        button_type="light",
                        width=200
                    ),
                    pn.widgets.Button(
                        name="🔆 Непрерывный шаблон",
                        button_type="light",
                        width=200
                    ),
                ),
                pn.pane.HTML("""
                <div style="color: #666; font-size: 0.9em; margin-top: 15px;">
                    <strong>💡 Быстрые шаблоны:</strong>
                    <ul>
                        <li>Импульсный: 8 импульсов, 10 Вт</li>
                        <li>Непрерывный: 5 Вт, 2 мс</li>
                    </ul>
                </div>
                """)
            ),
            title="⚡ Быстрые действия",
            collapsed=False,
            margin=(20, 0)
        )
        
        return pn.Column(
            pn.pane.HTML("<h3>🎯 Управление конфигурацией</h3>"),
            buttons_row,
            pn.Card(json_output, title="📝 JSON конфигурация", collapsed=False),
            quick_actions,
            sizing_mode='stretch_width'
        )
    
    # ============================================================================
    # ОБРАБОТЧИКИ СОБЫТИЙ
    # ============================================================================
    
    def calculate_parameters(self, event=None):
        """
        Рассчитывает ключевые параметры, необходимые для моделирования тепловых процессов, возникающих при лазерном воздействии на материал.
        
        Args:
            event (optional):  Event object, likely triggered by a UI element. Not used in the calculation itself. Defaults to None.
        
        Returns:
            None: The method updates the object's attributes with calculated values and updates the status message.
        """
        try:
            # Временные вычисления
            if self.laser_mode == 'continuous':
                peak_power = self.laser_continuous_power
                rep_rate = 1.0
                pulse_duration = 1e-6
            else:
                peak_power = self.laser_avg_power / (self.laser_rep_rate * self.laser_pulse_duration)
                rep_rate = self.laser_rep_rate
                pulse_duration = self.laser_pulse_duration
            
            # Характерные масштабы
            char_length = self.laser_beam_radius
            thermal_diffusivity = self.material_conductivity / (self.material_density * self.material_specific_heat)
            char_time = char_length**2 / thermal_diffusivity
            
            # Пиковая интенсивность
            peak_intensity = (2 * peak_power) / (np.pi * self.laser_beam_radius**2)
            
            # Характерная температура
            char_temp = ((1 - self.material_reflectivity) * 
                        peak_intensity * 
                        self.material_absorption * 
                        char_length**2 / 
                        self.material_conductivity)
            
            # Обновляем параметры
            self.calculated_char_length = f"{char_length*1e6:.2f}"
            self.calculated_char_time = f"{char_time*1e3:.2f}"
            self.calculated_char_temp = f"{char_temp:.1f}"
            self.calculated_peak_intensity = f"{peak_intensity/1e6:.1f}"
            
            self.update_status("✅ Параметры успешно рассчитаны")
            
        except Exception as e:
            self.update_status(f"❌ Ошибка расчета: {str(e)}", "danger")
    
    def calculate_initial_parameters(self):
        """
        Рассчитывает начальные параметры, необходимые для последующей обработки и моделирования.
        
        Args:
            self: Экземпляр класса LaserConfigApp, содержащий конфигурацию и данные.
        
        Returns:
            None: Метод изменяет состояние объекта, вычисляя и сохраняя параметры внутри него.
        """
        self.calculate_parameters()
    
    def generate_json(self, event=None):
        """
        Генерирует JSON конфигурацию на основе внутренних параметров приложения.
        
        Args:
            event (optional): Event object, not used in the current implementation. Defaults to None.
        
        Returns:
            None: The generated JSON configuration is stored in the `self.json_output` attribute.
                  Updates the application status to indicate success or failure.
        
        The method creates a JSON representation of the application's configuration, allowing for easy storage or transfer of settings. This is crucial for reproducibility and sharing of experimental setups or simulation parameters.
        """
        try:
            config = self.generate_config_dict()
            config_str = json.dumps(config, indent=2, default=str)
            self.json_output = config_str
            self.update_status("✅ JSON конфигурация сгенерирована")
        except Exception as e:
            self.update_status(f"❌ Ошибка генерации JSON: {str(e)}", "danger")
    
    def generate_config_dict(self):
        """
        Generates a dictionary containing all the necessary configuration parameters for the laser processing simulation.
        
        This dictionary encapsulates parameters related to the laser source, material properties, 
        PINN settings, and training configurations, providing a comprehensive setup for the simulation.
        The structure allows for easy access and modification of individual parameters.
        
        Args:
            self: An instance of the LaserConfigApp class, providing access to the configuration attributes.
        
        Returns:
            dict: A dictionary containing the complete configuration for the laser processing simulation, 
                  organized into 'laser', 'material', 'pinn', and 'training' sections.
        """
        return {
            "laser": {
                "wavelength": self.laser_wavelength,
                "rep_rate": self.laser_rep_rate if self.laser_mode == 'pulsed' else 1.0,
                "pulse_duration": self.laser_pulse_duration if self.laser_mode == 'pulsed' else 1e-6,
                "avg_power": self.laser_avg_power if self.laser_mode == 'pulsed' else 0.0,
                "beam_radius": self.laser_beam_radius,
                "scan_velocity": self.laser_scan_velocity,
                "mode": self.laser_mode,
                "continuous_power": self.laser_continuous_power,
                "num_pulses": self.num_pulses if self.laser_mode == 'pulsed' else 1,
                "simulation_time": self.simulation_time if self.laser_mode == 'continuous' else None
            },
            "material": {
                "density": self.material_density,
                "specific_heat": self.material_specific_heat,
                "conductivity": self.material_conductivity,
                "absorption": self.material_absorption,
                "reflectivity": self.material_reflectivity,
                "initial_temperature": self.initial_temperature
            },
            "pinn": {
                "num_gaussians": 1,
                "gaussian_spacing": 0.5,
                "sigma0": 0.1,
                "laser_amplitude": self.laser_amplitude,
                "collocation_points": {
                    "x": self.col_x,
                    "y": self.col_y,
                    "z": self.col_z,
                    "t": self.col_t
                },
                "visualization_points": {
                    "x": self.vis_x,
                    "y": self.vis_y,
                    "z": self.vis_z,
                    "t": self.vis_t
                }
            },
            "training": {
                "num_epochs": self.num_epochs,
                "learning_rate": self.learning_rate,
                "device": "auto",
                "loss_weights": {
                    "pde": self.loss_pde,
                    "ic": self.loss_ic,
                    "bc": self.loss_bc
                }
            }
        }
    
    def save_to_file(self, event=None):
        """
        Сохраняет текущую конфигурацию лазера в файл.
        
        Args:
            event (optional): Event object, likely from a GUI framework. Not used directly in the function's logic. Defaults to None.
        
        Returns:
            None
        
        The method ensures the configuration is in JSON format, generates a unique filename incorporating the laser mode, and then writes the JSON data to the file. This allows for persistent storage and retrieval of laser settings for later use or replication of experiments.
        """
        try:
            if not self.json_output:
                self.generate_json()
            
            filename = f"config_{self.laser_mode}_{np.random.randint(1000, 9999)}.json"
            filepath = Path(filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(self.json_output)
            
            self.update_status(f"✅ Конфигурация сохранена в файл: {filename}")
            
        except Exception as e:
            self.update_status(f"❌ Ошибка сохранения: {str(e)}", "danger")
    
    def reset_to_defaults(self, event=None):
        """
        Сбрасывает все параметры конфигурации к предопределенным значениям по умолчанию.
        
        Это необходимо для обеспечения воспроизводимости результатов и предоставления отправной точки для новых симуляций.
        Метод восстанавливает параметры лазера, материала, PINN и обучения к их начальным значениям,
        после чего пересчитывает производные параметры и обновляет статус приложения.
        
        Args:
            event (optional): Объект события, связанный с действием пользователя (например, нажатием кнопки).
                              Не используется напрямую в логике метода. Defaults to None.
        
        Returns:
            None
        """
        try:
            # Сброс параметров лазера
            self.laser_mode = 'continuous'
            self.laser_wavelength = 10.6e-6
            self.laser_beam_radius = 62e-6
            self.laser_scan_velocity = 0.06
            
            # Сброс параметров материала
            self.material_density = 2200.0
            self.material_specific_heat = 670.0
            self.material_conductivity = 1.4
            self.material_absorption = 5000.0
            self.material_reflectivity = 0.25
            self.initial_temperature = 300.0
            
            # Сброс параметров PINN
            self.laser_amplitude = 1.0
            self.col_x = 20
            self.col_y = 20
            self.col_z = 20
            self.col_t = 20
            self.vis_x = 30
            self.vis_y = 30
            self.vis_z = 30
            self.vis_t = 20
            
            # Сброс параметров обучения
            self.num_epochs = 1000
            self.learning_rate = 1e-3
            self.loss_pde = 1.0
            self.loss_ic = 1.0
            self.loss_bc = 2.0
            
            # Пересчет параметров
            self.calculate_parameters()
            self.update_status("✅ Все параметры сброшены к значениям по умолчанию")
            
        except Exception as e:
            self.update_status(f"❌ Ошибка сброса: {str(e)}", "danger")
    
    def apply_config(self, event=None):
        """
        Применяет текущую конфигурацию, генерируя словарь параметров и обновляя глобальные настройки для обеспечения согласованности системы.
        
        Args:
            event (optional): Event object, typically used in GUI frameworks. Defaults to None.
        
        Returns:
            None
        """
        try:
            # Создаем временный конфиг
            config_dict = self.generate_config_dict()
            
            # Обновляем глобальный объект конфигурации
            config.CONFIG.config = config.CONFIG.deep_update(config.CONFIG.config.copy(), config_dict)
            config.CONFIG.calculate_derived_parameters()
            
            self.update_status(f"✅ Конфигурация применена. Режим: {self.laser_mode}")
            
        except Exception as e:
            self.update_status(f"❌ Ошибка применения конфигурации: {str(e)}", "danger")
    
    def update_status(self, message, alert_type="info"):
        """
        Updates the status message displayed in the application.
        
        Args:
            message (str): The status message to display.
            alert_type (str, optional): The type of alert for the status message 
                ("info", "warning", "error", etc.). Defaults to "info".
        
        Returns:
            None
        """
        self.status_message = message
        self.status_bar.alert_type = alert_type
        self.status_bar.object = message
    
    def show(self):
        """
        Displays the main user interface layout.
        
        Args:
            self:  The LaserConfigApp instance.
        
        Returns:
            QtWidgets.QWidget: The main layout of the application, containing the user interface elements.
        """
        return self.layout

# ============================================================================
# ЗАПУСК ПРИЛОЖЕНИЯ
# ============================================================================

# Создаем экземпляр приложения
app = LaserConfigApp()

# Создаем панель с инструкцией
instructions = pn.pane.HTML("""
<div style="
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    padding: 25px;
    border-radius: 10px;
    color: white;
    margin: 20px 0;
">
    <h3 style="margin-top: 0;">🚀 Инструкция по использованию</h3>
    
    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px; margin: 15px 0;">
        <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px;">
            <div style="font-weight: bold; font-size: 1.1em;">1️⃣ Настройте параметры</div>
            <div style="opacity: 0.9; margin-top: 5px;">
                Используйте все вкладки для настройки параметров моделирования
            </div>
        </div>
        
        <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px;">
            <div style="font-weight: bold; font-size: 1.1em;">2️⃣ Рассчитайте</div>
            <div style="opacity: 0.9; margin-top: 5px;">
                Нажмите "Рассчитать все" для получения расчетных параметров
            </div>
        </div>
        
        <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px;">
            <div style="font-weight: bold; font-size: 1.1em;">3️⃣ Сгенерируйте JSON</div>
            <div style="opacity: 0.9; margin-top: 5px;">
                Создайте JSON конфигурацию для использования в моделировании
            </div>
        </div>
        
        <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius= 8px;">
            <div style="font-weight: bold; font-size: 1.1em;">4️⃣ Примените</div>
            <div style="opacity: 0.9; margin-top: 5px;">
                Нажмите "Применить" чтобы обновить глобальную конфигурацию
            </div>
        </div>
    </div>
    
    <div style="background: rgba(255,255,255,0.2); padding: 15px; border-radius: 8px; margin-top: 10px;">
        <strong>💡 Важно:</strong> После настройки обязательно нажмите "Применить" чтобы изменения вступили в силу!
    </div>
</div>
""")

# Собираем финальное приложение
final_app = pn.Column(
    app.title_pane,
    instructions,
    app.tabs,
    app.status_bar,
    sizing_mode='stretch_width',
    margin=(0, 20, 20, 20)
)

# ============================================================================
# СЕРВИРОВАНИЕ ПРИЛОЖЕНИЯ
# ============================================================================

if __name__ == "__main__":
    # Сохраняем как servable для запуска с помощью panel serve
    final_app.servable()
    
    # Для локального запуска с автоматическим открытием браузера
    # pn.serve(final_app, show=True, port=5006)
else:
    # Для использования в Jupyter Notebook
    final_app