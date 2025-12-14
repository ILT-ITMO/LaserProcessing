import numpy as np
import matplotlib.pyplot as plt
import config
from matplotlib.animation import FuncAnimation, PillowWriter
from conditions import convert_to_physical_coords, convert_to_physical_temperature, get_physical_extent
from matplotlib.patches import Circle, Polygon, Ellipse
from matplotlib.collections import PatchCollection

def visualize_laser_pulses():
    """
    Визуализация временного профиля лазерных импульсов.
    
    Метод отображает временную зависимость интенсивности лазерного излучения, 
    адаптируясь к текущему режиму работы лазера (импульсный или непрерывный).
    В импульсном режиме отображаются отдельные импульсы с указанием их параметров, 
    а в непрерывном - постоянный уровень интенсивности.
    
    Args:
        None
    
    Returns:
        None
    """
    # Получаем текущий режим из конфигурации
    current_mode = config.LASER_MODE
    
    if current_mode == "pulsed":
        # Для импульсного режима - гауссовы импульсы
        # Показываем первые несколько импульсов
        num_impulses_to_how = min(config.NUM_PULSES, 5)  # Максимум 5 импульсов для наглядности
        t_test = np.linspace(0, config.LASER_PULSE_PERIOD_NORM * num_impulses_to_how, 2000)
        source_values = np.zeros_like(t_test)
        
        for i, t_val in enumerate(t_test):
            t_mod = t_val % config.LASER_PULSE_PERIOD_NORM
            # Гауссов импульс
            source_values[i] = config.LASER_AMPLITUDE * np.exp(
                -(t_mod - config.LASER_PULSE_PERIOD_NORM/2)**2 / 
                (2 * config.LASER_PULSE_SIGMA_NORM**2)
            )
        
        # Конвертация в физическое время
        t_phys = t_test * config.CHARACTERISTIC_TIME * 1e6  # в микросекунды
        
        plt.figure(figsize=(12, 5))
        plt.plot(t_phys, source_values, 'r-', linewidth=2.5, label='Интенсивность лазера')
        plt.xlabel('Время (мкс)', fontsize=12)
        plt.ylabel('Относительная интенсивность', fontsize=12)
        
        title = f'Временной профиль лазерных импульсов (Импульсный режим)\n'
        title += f'Пиковая мощность: {config.LASER_PEAK_POWER:.1f} Вт, '
        title += f'Частота: {config.LASER_REP_RATE:.0f} Гц, '
        title += f'Длительность: {config.LASER_PULSE_DURATION*1e6:.1f} мкс'
        plt.title(title, fontsize=13)
        
        plt.grid(True, alpha=0.3)
        
        # Добавим вертикальные линии для периодов
        for i in range(num_impulses_to_how + 1):
            period_time = i * config.LASER_PULSE_PERIOD_NORM * config.CHARACTERISTIC_TIME * 1e6
            plt.axvline(x=period_time, color='gray', linestyle='--', alpha=0.5, linewidth=1)
            if i < num_impulses_to_how:
                plt.text(period_time + 0.5, 0.85 * config.LASER_AMPLITUDE, 
                        f'Импульс {i+1}', fontsize=10, ha='left')
        
        # Добавим заливку под кривой для первого импульса
        first_impulse_end = config.LASER_PULSE_PERIOD_NORM * config.CHARACTERISTIC_TIME * 1e6
        mask = t_phys <= first_impulse_end
        plt.fill_between(t_phys[mask], 0, source_values[mask], alpha=0.3, color='red')
        
        # Информационная панель
        info_text = f"Всего импульсов: {config.NUM_PULSES}\n"
        info_text += f"Период: {config.LASER_PULSE_PERIOD*1e6:.1f} мкс\n"
        info_text += f"Скважность: {config.LASER_DUTY_CYCLE*100:.1f}%\n"
        info_text += f"Средняя мощность: {config.LASER_AVG_POWER} Вт"
        
        plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
        
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig('animations/laser_pulse_profile_gaussian.png', dpi=150, bbox_inches='tight')
        plt.show()
        
    else:
        # Для непрерывного режима - постоянный сигнал
        t_test = np.linspace(0, config.SIMULATION_TIME_NORM, 1000)
        source_values = np.ones_like(t_test) * config.LASER_AMPLITUDE
        
        # Конвертация в физическое время
        t_phys = t_test * config.CHARACTERISTIC_TIME * 1e6  # в микросекунды
        
        plt.figure(figsize=(12, 5))
        plt.plot(t_phys, source_values, 'b-', linewidth=2.5, label='Интенсивность лазера')
        plt.fill_between(t_phys, 0, source_values, alpha=0.3, color='blue')
        plt.xlabel('Время (мкс)', fontsize=12)
        plt.ylabel('Относительная интенсивность', fontsize=12)
        
        title = 'Временной профиль лазерного излучения (Непрерывный режим)\n'
        title += f'Мощность: {config.LASER_CONTINUOUS_POWER} Вт, '
        title += f'Длина волны: {config.LASER_WAVELENGTH*1e6:.1f} мкм'
        plt.title(title, fontsize=13)
        
        plt.grid(True, alpha=0.3)
        plt.ylim(0, config.LASER_AMPLITUDE * 1.1)
        
        # Информационная панель
        info_text = f"Режим: НЕПРЕРЫВНЫЙ\n"
        info_text += f"Мощность: {config.LASER_CONTINUOUS_POWER} Вт\n"
        info_text += f"Интенсивность: {config.LASER_PEAK_INTENSITY/1e6:.2f} МВт/м²\n"
        info_text += f"Время моделирования: {config.SIMULATION_TIME_PHYSICAL*1e6:.1f} мкс"
        
        plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
        
        # Отметим начало и конец симуляции
        plt.axvline(x=0, color='green', linestyle='-', alpha=0.7, linewidth=2, label='Начало')
        plt.axvline(x=config.SIMULATION_TIME_PHYSICAL*1e6, color='red', linestyle='-', 
                   alpha=0.7, linewidth=2, label='Конец')
        
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig('animations/laser_continuous_profile.png', dpi=150, bbox_inches='tight')
        plt.show()

def visualize_laser_spatial_profile():
    """
    Визуализация пространственного профиля лазерного пучка.
    
    Метод генерирует двумерный контурный график и трехмерную поверхность, отображающие распределение интенсивности лазерного пучка в пространстве.  Визуализация включает информацию о радиусе пучка и режиме его работы (импульсный или непрерывный). Результат сохраняется в файл и отображается на экране.
    
    Args:
        None
    
    Returns:
        None
    """
    # Получаем текущий режим
    current_mode = config.LASER_MODE
    
    x_test = np.linspace(-1, 1, 200)
    y_test = np.linspace(-1, 1, 200)
    X, Y = np.meshgrid(x_test, y_test)
    
    # Гауссов профиль пучка
    spatial_dist = config.LASER_AMPLITUDE * np.exp(-(X**2 + Y**2) / (config.LASER_SIGMA_NORM**2))
    
    # Конвертация в физические координаты (мкм)
    X_phys = X * config.CHARACTERISTIC_LENGTH * 1e6
    Y_phys = Y * config.CHARACTERISTIC_LENGTH * 1e6
    
    fig = plt.figure(figsize=(15, 6))
    
    # 1. Контурный график
    ax1 = fig.add_subplot(1, 2, 1)
    contour = ax1.contourf(X_phys, Y_phys, spatial_dist, levels=50, cmap='hot')
    plt.colorbar(contour, ax=ax1, label='Относительная интенсивность')
    ax1.set_xlabel('x (мкм)', fontsize=11)
    ax1.set_ylabel('y (мкм)', fontsize=11)
    
    # Добавим перекрестие в центре
    ax1.axhline(y=0, color='white', linestyle='--', alpha=0.5, linewidth=1)
    ax1.axvline(x=0, color='white', linestyle='--', alpha=0.5, linewidth=1)
    ax1.plot(0, 0, 'w+', markersize=12, markeredgewidth=2, label='Центр пучка')
    
    # Добавим круг радиуса пучка
    beam_radius_um = config.LASER_BEAM_RADIUS * 1e6
    circle = plt.Circle((0, 0), beam_radius_um, fill=False, color='cyan', 
                       linestyle='-', linewidth=2, alpha=0.8, label=f'Радиус пучка: {beam_radius_um:.0f} мкм')
    ax1.add_patch(circle)
    
    # Добавим круг на 1/e² радиусе
    radius_1e2 = beam_radius_um / np.sqrt(2)
    circle_1e2 = plt.Circle((0, 0), radius_1e2, fill=False, color='lime', 
                           linestyle='--', linewidth=1.5, alpha=0.7, label=f'1/e² радиус: {radius_1e2:.0f} мкм')
    ax1.add_patch(circle_1e2)
    
    mode_text = "ИМПУЛЬСНЫЙ" if current_mode == "pulsed" else "НЕПРЕРЫВНЫЙ"
    power_text = f"{config.LASER_AVG_POWER} Вт (ср.)" if current_mode == "pulsed" else f"{config.LASER_CONTINUOUS_POWER} Вт"
    
    ax1.set_title(f'Пространственное распределение лазерного пучка\nРежим: {mode_text}', fontsize=13)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.2)
    ax1.set_aspect('equal')
    
    # 2. 3D поверхность
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    
    # Берем подмножество точек для 3D визуализации
    X_3d = X_phys[::4, ::4]
    Y_3d = Y_phys[::4, ::4]
    Z_3d = spatial_dist[::4, ::4]
    
    surf = ax2.plot_surface(X_3d, Y_3d, Z_3d, cmap='hot', alpha=0.8, 
                           rstride=1, cstride=1, linewidth=0.5, antialiased=True)
    ax2.set_xlabel('x (мкм)', fontsize=11)
    ax2.set_ylabel('y (мкм)', fontsize=11)
    ax2.set_zlabel('Интенсивность', fontsize=11)
    ax2.set_title('3D профиль интенсивности', fontsize=13)
    
    # Добавим информацию о режиме в 3D график
    info_text_3d = f"Режим: {mode_text}\n"
    info_text_3d += f"Мощность: {power_text}\n"
    info_text_3d += f"Радиус: {beam_radius_um:.0f} мкм"
    
    ax2.text2D(0.05, 0.95, info_text_3d, transform=ax2.transAxes,
              fontsize=10, verticalalignment='top',
              bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
    
    # Настройка угла обзора
    ax2.view_init(elev=30, azim=45)
    
    plt.suptitle(f'Характеристики лазерного пучка\n', fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig(f'animations/laser_spatial_profile_{current_mode}.png', dpi=150, bbox_inches='tight')
    plt.show()

def add_isotherm_plot(ax, U_physical, x_phys, y_phys, z_phys, time_idx, 
                     isotherm_temp=1900, mode_text="", power_text="", 
                     beam_radius_um=None):
    """
    Plots an isotherm on the depth and width of the simulation.
    
    Args:
        ax: The axes object to draw on.
        U_physical: The physical temperature field.
        x_phys: Physical x coordinates (µm).
        y_phys: Physical y coordinates (µm).
        z_phys: Physical z coordinates (depth, µm).
        time_idx: The time index.
        isotherm_temp: The temperature of the isotherm (K). Defaults to 1900.
        mode_text: Text describing the mode. Defaults to "".
        power_text: Text describing the power. Defaults to "".
        beam_radius_um: The beam radius in µm. Defaults to None.
    
    Returns:
        None
    """
    # Центральные индексы
    center_x = len(x_phys) // 2
    center_y = len(y_phys) // 2
    
    # Изотерма на поверхности (z = max)
    surface_z_idx = -1
    temp_surface = U_physical[:, :, surface_z_idx, time_idx]
    
    # Создаем контуры для изотермы
    X_surf, Y_surf = np.meshgrid(x_phys, y_phys)
    
    # Находим контур изотермы
    if np.max(temp_surface) >= isotherm_temp:
        contour_levels = [isotherm_temp]
        cs = ax.contour(X_surf, Y_surf, temp_surface.T, levels=contour_levels, 
                       colors='white', linewidths=2.5, linestyles='-', alpha=0.9)
        
        # Если контур найден, закрашиваем внутреннюю область
        # Используем новый API для matplotlib
        paths = cs.get_paths()
        if len(paths) > 0:
            for path in paths:
                vertices = path.vertices
                if len(vertices) > 2:
                    # Закрашиваем область внутри контура
                    polygon = Polygon(vertices, closed=True, fill=True, 
                                     facecolor='lime', alpha=0.3, 
                                     edgecolor='lime', linewidth=1)
                    ax.add_patch(polygon)
                    
                    # Подписываем изотерму
                    if len(vertices) > 10:
                        # Берем точку примерно посередине контура
                        mid_idx = len(vertices) // 2
                        x_mid, y_mid = vertices[mid_idx]
                        ax.text(x_mid, y_mid, f'{isotherm_temp} K', 
                               color='lime', fontsize=9, fontweight='bold',
                               ha='center', va='center',
                               bbox=dict(boxstyle="round,pad=0.2", 
                                        facecolor="black", alpha=0.5))
    
    # Добавляем профиль по глубине в центре
    temp_profile = U_physical[center_x, center_y, :, time_idx]
    
    # Находим глубину, где температура достигает isotherm_temp
    depth_at_isotherm = None
    for i in range(len(z_phys)-1, -1, -1):
        if temp_profile[i] >= isotherm_temp:
            depth_at_isotherm = z_phys[i]
            break
    
    # Находим ширину на поверхности, где температура достигает isotherm_temp
    width_at_isotherm = None
    if np.max(temp_surface) >= isotherm_temp:
        # Проходим от центра к краю
        for i in range(center_x, len(x_phys)):
            if temp_surface[i, center_y] < isotherm_temp:
                width_at_isotherm = abs(x_phys[i])
                break
    
    # Добавляем информацию об изотерме
    info_text = f"ИЗОТЕРМА {isotherm_temp} K:\n"
    if depth_at_isotherm is not None:
        info_text += f"Глубина: {depth_at_isotherm:.1f} мкм\n"
    else:
        info_text += f"Глубина: не достигнута\n"
    
    if width_at_isotherm is not None:
        info_text += f"Ширина: {width_at_isotherm:.1f} мкм"
    else:
        info_text += f"Ширина: не достигнута"
    
    # Добавляем текстовую информацию
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7,
                     edgecolor='lime'))
    
    # Добавляем перекрестие в центре
    ax.axhline(y=0, color='white', linestyle='--', alpha=0.5, linewidth=1)
    ax.axvline(x=0, color='white', linestyle='--', alpha=0.5, linewidth=1)
    ax.plot(0, 0, 'w+', markersize=10, markeredgewidth=1.5)
    
    # Добавляем круг радиуса пучка если указан
    if beam_radius_um is not None:
        circle = Circle((0, 0), beam_radius_um, fill=False, color='cyan', 
                       linestyle='-', linewidth=1.5, alpha=0.7)
        ax.add_patch(circle)
    
    # Заголовок
    ax.set_title(f'Изотерма {isotherm_temp} K на поверхности\nГлубина/Ширина зоны нагрева', fontsize=11)
    ax.set_xlabel('x (мкм)', fontsize=10)
    ax.set_ylabel('y (мкм)', fontsize=10)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

def create_animation(U_data, x_plot, y_plot, z_plot, t_plot, title, filename):
    """
    Создает анимацию, визуализирующую распределение температуры в трехмерном пространстве, 
    полученном в результате моделирования лазерного воздействия на материал. 
    Анимация отображает срезы температуры в различных плоскостях и на разных глубинах, 
    что позволяет анализировать динамику нагрева и теплопроводности.
    Поддерживает визуализацию как для импульсного, так и для непрерывного режимов лазерного излучения.
    """
    # Конвертация в физические величины
    U_physical = convert_to_physical_temperature(U_data)
    (x_phys_min, x_phys_max), (y_phys_min, y_phys_max), (z_phys_min, z_phys_max) = get_physical_extent(
        [x_plot[0], x_plot[-1]], [y_plot[0], y_plot[-1]], [z_plot[0], z_plot[-1]]
    )
    
    # Создаем массивы физических координат
    x_phys = np.linspace(x_phys_min, x_phys_max, len(x_plot))
    y_phys = np.linspace(y_phys_min, y_phys_max, len(y_plot))
    z_phys = np.linspace(z_phys_min, z_phys_max, len(z_plot))
    
    t_phys = t_plot * config.CHARACTERISTIC_TIME * 1e6  # мкс
    
    # Определяем индексы для срезов на глубинах 10, 20, 30 мкм
    z_phys_values = np.array(z_plot) * config.CHARACTERISTIC_LENGTH * 1e6  # все глубины в мкм
    target_depths = [10, 20, 30]  # целевые глубины в мкм
    
    depth_indices = []
    actual_depths = []
    for depth in target_depths:
        idx = np.argmin(np.abs(z_phys_values - depth))
        depth_indices.append(idx)
        actual_depths.append(z_phys_values[idx])
    
    fig = plt.figure(figsize=(25, 14))  # Увеличили высоту для нового графика
    
    def update(frame):
        fig.clear()
        
        current_time_phys = t_phys[frame]
        
        # 1. XY срез (поверхность)
        ax1 = fig.add_subplot(3, 5, 1)
        slice_idx_xy = len(z_plot) - 1  # Поверхность (z = max)
        data_xy = U_physical[:, :, slice_idx_xy, frame].T
        
        im1 = ax1.imshow(data_xy, extent=[x_phys_min, x_phys_max, y_phys_min, y_phys_max], 
                        origin='lower', aspect='auto', cmap='hot', 
                        vmin=config.INITIAL_TEMPERATURE, vmax=np.max(U_physical))
        ax1.set_title('XY срез (поверхность)', fontsize=11)
        ax1.set_xlabel('x (мкм)', fontsize=10)
        ax1.set_ylabel('y (мкм)', fontsize=10)
        
        # Добавим перекрестие в центре
        ax1.axhline(y=0, color='white', linestyle='--', alpha=0.5, linewidth=1)
        ax1.axvline(x=0, color='white', linestyle='--', alpha=0.5, linewidth=1)
        ax1.plot(0, 0, 'w+', markersize=10, markeredgewidth=1.5)
        
        # Добавим круг радиуса пучка
        beam_radius_um = config.LASER_BEAM_RADIUS * 1e6
        circle = Circle((0, 0), beam_radius_um, fill=False, color='cyan', 
                       linestyle='-', linewidth=1.5, alpha=0.7)
        ax1.add_patch(circle)
        
        plt.colorbar(im1, ax=ax1, shrink=0.8, label='Температура (K)')
        
        # 2. XZ срез
        ax2 = fig.add_subplot(3, 5, 2)
        slice_idx_xz = len(y_plot) // 2
        data_xz = U_physical[:, slice_idx_xz, :, frame].T
        
        im2 = ax2.imshow(data_xz, extent=[x_phys_min, x_phys_max, z_phys_min, z_phys_max], 
                        origin='lower', aspect='auto', cmap='hot',
                        vmin=config.INITIAL_TEMPERATURE, vmax=np.max(U_physical))
        ax2.set_title('XZ срез (центральный)', fontsize=11)
        ax2.set_xlabel('x (мкм)', fontsize=10)
        ax2.set_ylabel('z (мкм)', fontsize=10)
        
        # Добавляем линии на целевых глубинах
        for depth, color in zip(actual_depths, ['cyan', 'lime', 'yellow']):
            ax2.axhline(y=depth, color=color, linestyle='--', alpha=0.7, linewidth=1)
        
        # Вертикальная линия в центре
        ax2.axvline(x=0, color='white', linestyle='--', alpha=0.5, linewidth=1)
        plt.colorbar(im2, ax=ax2, shrink=0.8, label='Температура (K)')
        
        # 3. YZ срез
        ax3 = fig.add_subplot(3, 5, 3)
        slice_idx_yz = len(x_plot) // 2
        data_yz = U_physical[slice_idx_yz, :, :, frame].T
        
        im3 = ax3.imshow(data_yz, extent=[y_phys_min, y_phys_max, z_phys_min, z_phys_max], 
                        origin='lower', aspect='auto', cmap='hot',
                        vmin=config.INITIAL_TEMPERATURE, vmax=np.max(U_physical))
        ax3.set_title('YZ срез (центральный)', fontsize=11)
        ax3.set_xlabel('y (мкм)', fontsize=10)
        ax3.set_ylabel('z (мкм)', fontsize=10)
        
        # Добавляем линии на целевых глубинах
        for depth, color in zip(actual_depths, ['cyan', 'lime', 'yellow']):
            ax3.axhline(y=depth, color=color, linestyle='--', alpha=0.7, linewidth=1)
        
        # Вертикальная линия в центре
        ax3.axvline(x=0, color='white', linestyle='--', alpha=0.5, linewidth=1)
        plt.colorbar(im3, ax=ax3, shrink=0.8, label='Температура (K)')
        
        # 4-6. XY срезы на целевых глубинах
        colors = ['cyan', 'lime', 'yellow']
        for i, (depth_idx, depth, color) in enumerate(zip(depth_indices, actual_depths, colors)):
            ax = fig.add_subplot(3, 5, 4 + i)
            data_depth = U_physical[:, :, depth_idx, frame].T
            
            im = ax.imshow(data_depth, extent=[x_phys_min, x_phys_max, y_phys_min, y_phys_max], 
                          origin='lower', aspect='auto', cmap='hot',
                          vmin=config.INITIAL_TEMPERATURE, vmax=np.max(U_physical))
            ax.set_title(f'XY срез (z = {depth:.1f} мкм)', fontsize=11)
            ax.set_xlabel('x (мкм)', fontsize=10)
            ax.set_ylabel('y (мкм)', fontsize=10)
            
            # Добавляем перекрестие в центре
            ax.axhline(y=0, color='white', linestyle='--', alpha=0.5, linewidth=1)
            ax.axvline(x=0, color='white', linestyle='--', alpha=0.5, linewidth=1)
            
            # Добавляем рамку соответствующего цвета
            for spine in ax.spines.values():
                spine.set_color(color)
                spine.set_linewidth(2)
            
            plt.colorbar(im, ax=ax, shrink=0.8, label='Температура (K)')
        
        # 7. График с изотермой 1900 K
        ax7 = fig.add_subplot(3, 5, 7)
        
        # Получаем текст режима и мощности
        mode_text = "ИМПУЛЬСНЫЙ" if config.LASER_MODE == "pulsed" else "НЕПРЕРЫВНЫЙ"
        power_text = f"{config.LASER_AVG_POWER} Вт (ср.)" if config.LASER_MODE == "pulsed" else f"{config.LASER_CONTINUOUS_POWER} Вт"
        
        # Добавляем график с изотермой
        add_isotherm_plot(ax7, U_physical, x_phys, y_phys, z_phys, frame, 
                         isotherm_temp=1900, mode_text=mode_text, power_text=power_text,
                         beam_radius_um=beam_radius_um)
        
        # 8. Временной профиль лазера (физическое время)
        ax8 = fig.add_subplot(3, 5, 8)
        
        if config.LASER_MODE == "pulsed":
            # Временной диапазон для всех импульсов
            t_range_norm = np.linspace(0, config.SIMULATION_TIME_NORM, 1000)
            t_range_phys = t_range_norm * config.CHARACTERISTIC_TIME * 1e6  # мкс
            laser_profile = np.zeros_like(t_range_norm)
            
            for i, t_val in enumerate(t_range_norm):
                t_mod = t_val % config.LASER_PULSE_PERIOD_NORM
                # Гауссов импульс
                laser_profile[i] = config.LASER_AMPLITUDE * np.exp(
                    -(t_mod - config.LASER_PULSE_PERIOD_NORM/2)**2 / 
                    (2 * config.LASER_PULSE_SIGMA_NORM**2))
            
            ax8.plot(t_range_phys, laser_profile, 'r-', linewidth=1.5)
            ax8.axvline(x=current_time_phys, color='blue', linestyle='--', alpha=0.8, linewidth=2)
            
            # Отметим все импульсы вертикальными линиями
            for i in range(config.NUM_PULSES + 1):
                impulse_time = i * config.LASER_PULSE_PERIOD_NORM * config.CHARACTERISTIC_TIME * 1e6
                ax8.axvline(x=impulse_time, color='gray', linestyle=':', alpha=0.5, linewidth=0.8)
                if i < config.NUM_PULSES:
                    ax8.text(impulse_time + 2, 0.8, f'{i+1}', fontsize=8, ha='left')
            
            ax8.set_title(f'Лазерные импульсы ({config.NUM_PULSES} импульсов)', fontsize=11)
            ax8.set_ylim(0, 1.1)
            
        else:
            # Непрерывный режим - постоянный сигнал
            t_range_norm = np.linspace(0, config.SIMULATION_TIME_NORM, 1000)
            t_range_phys = t_range_norm * config.CHARACTERISTIC_TIME * 1e6
            laser_profile = np.ones_like(t_range_norm) * config.LASER_AMPLITUDE
            
            ax8.plot(t_range_phys, laser_profile, 'b-', linewidth=2)
            ax8.axvline(x=current_time_phys, color='red', linestyle='--', alpha=0.8, linewidth=2)
            ax8.fill_between(t_range_phys, 0, laser_profile, alpha=0.3, color='blue')
            
            ax8.set_title('Непрерывный лазерный источник', fontsize=11)
            ax8.set_ylim(0, config.LASER_AMPLITUDE * 1.1)
        
        ax8.set_xlabel('Время (мкс)', fontsize=10)
        ax8.set_ylabel('Интенсивность', fontsize=10)
        ax8.grid(True, alpha=0.3)
        ax8.set_xlim(0, max(t_phys))
        
        # 9. Информация о системе
        ax9 = fig.add_subplot(3, 5, 9)
        ax9.axis('off')
        
        if config.LASER_MODE == "pulsed":
            pulse_number = min(int(t_plot[frame] // config.LASER_PULSE_PERIOD_NORM) + 1, config.NUM_PULSES)
            time_in_pulse_norm = t_plot[frame] % config.LASER_PULSE_PERIOD_NORM
            time_in_pulse_phys = time_in_pulse_norm * config.CHARACTERISTIC_TIME * 1e6
        else:
            pulse_number = 1
            time_in_pulse_norm = t_plot[frame]
            time_in_pulse_phys = current_time_phys
        
        max_temp = np.max(U_physical[:, :, :, frame])
        min_temp = np.min(U_physical[:, :, :, frame])
        
        # Температуры на разных глубинах
        temp_at_depths = []
        for depth_idx, depth in zip(depth_indices, actual_depths):
            temp = np.max(U_physical[:, :, depth_idx, frame])
            temp_at_depths.append((depth, temp))
        
        # Проверяем достижение изотермы 1900 K
        isotherm_reached = max_temp >= 1900
        isotherm_status = "ДА" if isotherm_reached else "НЕТ"
        isotherm_color = "lime" if isotherm_reached else "red"
        
        info_text = f"ВРЕМЯ: {current_time_phys:.1f} мкс\n"
        info_text += "=" * 30 + "\n"
        
        if config.LASER_MODE == "pulsed":
            info_text += f"Импульс: {pulse_number}/{config.NUM_PULSES}\n"
            info_text += f"В импульсе: {time_in_pulse_phys:.1f} мкс\n"
        else:
            info_text += f"Режим: НЕПРЕРЫВНЫЙ\n"
            info_text += f"Мощность: {config.LASER_CONTINUOUS_POWER} Вт\n"
        
        info_text += f"\nТЕМПЕРАТУРА:\n"
        info_text += f"Макс: {max_temp:.1f} K\n"
        info_text += f"Мин: {min_temp:.1f} K\n"
        info_text += f"Перегрев: {max_temp - config.INITIAL_TEMPERATURE:.1f} K\n"
        info_text += f"Начальная: {config.INITIAL_TEMPERATURE} K\n"
        
        info_text += f"\nИЗОТЕРМА 1900 K:\n"
        info_text += f"Достигнута: {isotherm_status}\n"
        
        info_text += f"\nНА ГЛУБИНАХ:\n"
        for depth, temp in temp_at_depths:
            info_text += f"  {depth:.1f} мкм: {temp:.1f} K\n"
        
        if config.LASER_MODE == "pulsed":
            if time_in_pulse_norm <= config.LASER_PULSE_DURATION_NORM * 2:
                info_text += "\nЛазер: АКТИВЕН"
            else:
                info_text += "\nЛазер: ВЫКЛ"
        else:
            info_text += "\nЛазер: ПОСТОЯННО ВКЛ"
        
        ax9.text(0.1, 0.5, info_text, fontsize=10, va='center', linespacing=1.4,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        # 10. Пространственный профиль лазера (физические координаты)
        ax10 = fig.add_subplot(3, 5, 11)
        x_profile_norm = np.linspace(-1, 1, 100)
        x_profile_phys = x_profile_norm * config.CHARACTERISTIC_LENGTH * 1e6
        laser_spatial = config.LASER_AMPLITUDE * np.exp(-x_profile_norm**2 / (config.LASER_SIGMA_NORM**2))
        
        ax10.plot(x_profile_phys, laser_spatial, 'g-', linewidth=2.5)
        ax10.set_xlabel('x (мкм)', fontsize=10)
        ax10.set_ylabel('Интенсивность', fontsize=10)
        
        beam_radius_um = config.LASER_BEAM_RADIUS * 1e6
        ax10.set_title(f'Пространственный профиль пучка\n(радиус {beam_radius_um:.0f} мкм)', fontsize=11)
        
        ax10.grid(True, alpha=0.3)
        ax10.set_xlim(x_phys_min, x_phys_max)
        ax10.set_ylim(0, 1.1)
        
        # Отметим радиус пучка
        ax10.axvline(x=beam_radius_um, color='cyan', linestyle='--', alpha=0.7, linewidth=1)
        ax10.axvline(x=-beam_radius_um, color='cyan', linestyle='--', alpha=0.7, linewidth=1)
        ax10.text(beam_radius_um, 0.5, f'{beam_radius_um:.0f} мкм', 
                fontsize=9, color='cyan', ha='left')
        
        # 11. График температуры по глубине
        ax11 = fig.add_subplot(3, 5, 12)
        center_x = len(x_plot) // 2
        center_y = len(y_plot) // 2
        temp_vs_depth = U_physical[center_x, center_y, :, frame]
        
        ax11.plot(temp_vs_depth, z_phys_values, 'b-', linewidth=2.5)
        ax11.set_xlabel('Температура (K)', fontsize=10)
        ax11.set_ylabel('Глубина z (мкм)', fontsize=10)
        ax11.set_title('Температура по глубине\n(в центре пучка)', fontsize=11)
        ax11.grid(True, alpha=0.3)
        ax11.set_ylim(z_phys_min, z_phys_max)
        
        # Добавляем горизонтальную линию для изотермы 1900 K
        ax11.axhline(y=0, color='lime', linestyle='-', alpha=0.5, linewidth=1)
        ax11.axvline(x=1900, color='lime', linestyle='-', alpha=0.5, linewidth=1, 
                    label=f'Изотерма 1900 K')
        
        # Добавляем маркеры на целевых глубинах
        for depth, color in zip(actual_depths, ['cyan', 'lime', 'yellow']):
            depth_idx = np.argmin(np.abs(z_phys_values - depth))
            temp = temp_vs_depth[depth_idx]
            ax11.plot(temp, depth, 'o', color=color, markersize=8, markeredgecolor='black')
            ax11.text(temp + 5, depth, f'{temp:.0f}K', fontsize=9, va='center')
        
        ax11.legend(fontsize=9)
        
        # 12. Дополнительный график: профиль температуры вдоль X на поверхности
        ax12 = fig.add_subplot(3, 5, 13)
        y_center_idx = len(y_plot) // 2
        x_profile = U_physical[:, y_center_idx, -1, frame]
        
        ax12.plot(x_phys, x_profile, 'r-', linewidth=2.5, label='Поверхность')
        ax12.fill_between(x_phys, config.INITIAL_TEMPERATURE, x_profile, alpha=0.2, color='red')
        
        # Добавляем горизонтальную линию для изотермы 1900 K
        ax12.axhline(y=1900, color='lime', linestyle='-', alpha=0.7, linewidth=2, 
                    label=f'Изотерма 1900 K')
        
        ax12.set_xlabel('x (мкм)', fontsize=10)
        ax12.set_ylabel('Температура (K)', fontsize=10)
        ax12.set_title('Профиль температуры вдоль X\n(поверхность, y=0)', fontsize=11)
        ax12.grid(True, alpha=0.3)
        ax12.legend(fontsize=9)
        
        # 13. График ширины зоны нагрева выше 1900 K
        ax13 = fig.add_subplot(3, 5, 14)
        
        # Рассчитываем ширину зоны нагрева выше 1900 K на поверхности
        width_1900 = None
        depth_1900 = None
        
        # Ширина на поверхности
        surface_temp = U_physical[:, center_y, -1, frame]
        left_idx = None
        right_idx = None
        
        for i in range(center_x, len(x_phys)):
            if surface_temp[i] >= 1900 and right_idx is None:
                right_idx = i
            if surface_temp[center_x - (i-center_x)] >= 1900 and left_idx is None:
                left_idx = center_x - (i-center_x)
        
        if left_idx is not None and right_idx is not None:
            width_1900 = x_phys[right_idx] - x_phys[left_idx]
        
        # Глубина проникновения 1900 K
        center_temp_profile = U_physical[center_x, center_y, :, frame]
        for i in range(len(z_phys)):
            if center_temp_profile[i] >= 1900:
                depth_1900 = z_phys[i]
                break
        
        # Создаем простую визуализацию
        if width_1900 is not None and depth_1900 is not None:
            # Рисуем эллипс зоны нагрева
            ellipse = Ellipse((0, -depth_1900/2), width=width_1900, height=depth_1900,
                             edgecolor='lime', facecolor='lime', alpha=0.3, 
                             linewidth=2, label=f'Зона >1900 K')
            ax13.add_patch(ellipse)
            
            # Подписи
            ax13.text(0, -depth_1900/2, f'Ш: {width_1900:.1f} мкм\nГ: {depth_1900:.1f} мкм',
                     ha='center', va='center', fontsize=10, fontweight='bold',
                     bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
        
        ax13.set_xlim(-beam_radius_um*2, beam_radius_um*2)
        ax13.set_ylim(-z_phys_max, 0)
        ax13.set_xlabel('Ширина (мкм)', fontsize=10)
        ax13.set_ylabel('Глубина (мкм)', fontsize=10)
        ax13.set_title('Зона нагрева >1900 K', fontsize=11)
        ax13.grid(True, alpha=0.3)
        ax13.set_aspect('equal')
        ax13.invert_yaxis()  # Глубина увеличивается вниз
        
        if width_1900 is not None and depth_1900 is not None:
            ax13.legend(fontsize=9)
        
        # 15. Пустая область для баланса или дополнительной информации
        ax15 = fig.add_subplot(3, 5, 15)
        ax15.axis('off')
        
        # Добавляем сводную информацию об изотерме
        summary_text = "СВОДКА ПО ИЗОТЕРМЕ 1900 K\n"
        summary_text += "=" * 25 + "\n\n"
        
        if width_1900 is not None:
            summary_text += f"• Ширина зоны: {width_1900:.1f} мкм\n"
        else:
            summary_text += "• Ширина зоны: не достигнута\n"
        
        if depth_1900 is not None:
            summary_text += f"• Глубина проникновения: {depth_1900:.1f} мкм\n"
        else:
            summary_text += "• Глубина проникновения: не достигнута\n"
        
        summary_text += f"• Макс. температура: {max_temp:.1f} K\n"
        summary_text += f"• Перегрев: {max_temp - config.INITIAL_TEMPERATURE:.1f} K\n"
        
        # Рассчитываем отношение ширины к глубине
        if width_1900 is not None and depth_1900 is not None:
            aspect_ratio = width_1900 / depth_1900
            summary_text += f"• Отношение Ш/Г: {aspect_ratio:.2f}"
        
        ax15.text(0.1, 0.5, summary_text, fontsize=10, va='center', linespacing=1.5,
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
        
        # Супер-заголовок
        mode_text = "ИМПУЛЬСНЫЙ" if config.LASER_MODE == "pulsed" else "НЕПРЕРЫВНЫЙ"
        suptitle = f'{title}\nРежим: {mode_text}, Время: {current_time_phys:.1f} мкс'
        if config.LASER_MODE == "pulsed":
            suptitle += f' (Импульс {pulse_number}/{config.NUM_PULSES})'
        
        plt.suptitle(suptitle, fontsize=14, y=0.98)
        plt.tight_layout()
        
        return fig,

    ani = FuncAnimation(fig, update, frames=len(t_plot), interval=500, blit=False, repeat=True)
    
    writer = PillowWriter(fps=2)
    ani.save(filename, writer=writer, dpi=100)
    plt.close(fig)
    print(f"Анимация сохранена: {filename}")
    return ani

def plot_temperature_evolution(U_data, x_plot, y_plot, z_plot, t_plot):
    """
    Plots the temperature evolution at the center of the laser beam over time.
    
    This visualization helps to understand the heating dynamics within the material 
    during laser exposure, accommodating both pulsed and continuous laser modes. 
    The plot displays the temperature changes, key temperature thresholds, and 
    relevant parameters like pulse timings or continuous power levels.
    
    Args:
        U_data (numpy.ndarray): Temperature data.
        x_plot (numpy.ndarray): X-coordinates for plotting.
        y_plot (numpy.ndarray): Y-coordinates for plotting.
        z_plot (numpy.ndarray): Z-coordinates for plotting.
        t_plot (numpy.ndarray): Time points for plotting.
    
    Returns:
        numpy.ndarray: Temperature values at the center of the beam over time.
    """
    U_physical = convert_to_physical_temperature(U_data)
    t_phys = t_plot * config.CHARACTERISTIC_TIME * 1e6  # мкс
    
    # Температура в центре пучка на поверхности
    center_x = len(x_plot) // 2
    center_y = len(y_plot) // 2
    surface_z = len(z_plot) - 1
    
    center_temperature = U_physical[center_x, center_y, surface_z, :]
    
    plt.figure(figsize=(14, 7))
    
    # Основной график температуры
    plt.plot(t_phys, center_temperature, 'b-', linewidth=3, label='Температура в центре пучка')
    
    # Начальная температура
    plt.axhline(y=config.INITIAL_TEMPERATURE, color='gray', linestyle='--', linewidth=2,
                label=f'Начальная температура ({config.INITIAL_TEMPERATURE} K)')
    
    # Изотерма 1900 K
    plt.axhline(y=1900, color='lime', linestyle='-', linewidth=2, alpha=0.7,
                label='Изотерма 1900 K')
    
    if config.LASER_MODE == "pulsed":
        # Отметим моменты импульсов
        impulse_colors = plt.cm.Reds(np.linspace(0.3, 0.8, config.NUM_PULSES))
        
        for i in range(config.NUM_PULSES):
            impulse_time = i * config.LASER_PULSE_PERIOD_NORM * config.CHARACTERISTIC_TIME * 1e6
            impulse_end = impulse_time + config.LASER_PULSE_DURATION_NORM * config.CHARACTERISTIC_TIME * 1e6
            
            # Заливка области импульса
            plt.axvspan(impulse_time, impulse_end, alpha=0.15, color=impulse_colors[i])
            
            # Вертикальные линии
            plt.axvline(x=impulse_time, color='red', linestyle=':', alpha=0.6, linewidth=1)
            if i == 0:
                plt.axvline(x=impulse_end, color='red', linestyle=':', alpha=0.6, linewidth=1, 
                           label='Границы импульсов')
            
            # Номера импульсов
            if i < min(config.NUM_PULSES, 10):  # Не более 10 подписей
                plt.text(impulse_time + 2, np.min(center_temperature) + 5, f'{i+1}', 
                        fontsize=9, color='red', fontweight='bold')
        
        title_text = f'Эволюция температуры в центре лазерного пучка\n'
        title_text += f'Импульсный режим: {config.NUM_PULSES} импульсов, '
        title_text += f'{config.LASER_AVG_POWER} Вт (ср.), {config.LASER_REP_RATE:.0f} Гц'
        
        # Добавим легенду для импульсов
        from matplotlib.patches import Patch
        impulse_patch = Patch(facecolor='red', alpha=0.15, label='Область импульсов')
        
    else:
        # Непрерывный режим
        title_text = f'Эволюция температуры в центре лазерного пучка\n'
        title_text += f'Непрерывный режим: {config.LASER_CONTINUOUS_POWER} Вт, '
        title_text += f'Время моделирования: {config.SIMULATION_TIME_PHYSICAL*1e6:.1f} мкс'
        
        # Заливка области нагрева
        plt.fill_between(t_phys, config.INITIAL_TEMPERATURE, center_temperature, 
                        alpha=0.2, color='blue', label='Область нагрева')
    
    plt.xlabel('Время (мкс)', fontsize=12)
    plt.ylabel('Температура (K)', fontsize=12)
    plt.title(title_text, fontsize=13)
    plt.grid(True, alpha=0.3)
    
    # Добавим информацию о максимальной температуре
    max_temp = np.max(center_temperature)
    max_time = t_phys[np.argmax(center_temperature)]
    overheating = max_temp - config.INITIAL_TEMPERATURE
    
    # Проверяем достижение изотермы 1900 K
    isotherm_reached = np.any(center_temperature >= 1900)
    isotherm_time = None
    if isotherm_reached:
        isotherm_idx = np.where(center_temperature >= 1900)[0][0]
        isotherm_time = t_phys[isotherm_idx]
    
    info_text = f"Макс. температура: {max_temp:.1f} K\n"
    info_text += f"Перегрев: {overheating:.1f} K\n"
    info_text += f"Время макс. нагрева: {max_time:.1f} мкс\n"
    
    if isotherm_reached:
        info_text += f"Изотерма 1900 K достигнута\n"
        info_text += f"Время достижения: {isotherm_time:.1f} мкс"
    else:
        info_text += f"Изотерма 1900 K не достигнута"
    
    plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes,
             fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
    
    # Отметим точку максимальной температуры
    plt.scatter([max_time], [max_temp], color='red', s=100, zorder=5, 
                label=f'Максимум: {max_temp:.1f} K')
    
    # Отметим точку достижения изотермы 1900 K если достигнута
    if isotherm_reached:
        plt.scatter([isotherm_time], [1900], color='lime', s=100, zorder=5,
                    edgecolor='black', linewidth=2, label=f'Изотерма 1900 K')
    
    plt.legend(loc='lower right', fontsize=10)
    plt.tight_layout()
    
    mode = config.LASER_MODE
    plt.savefig(f'animations/temperature_evolution_center_{mode}.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return center_temperature

def plot_depth_temperature_profiles(U_data, x_plot, y_plot, z_plot, t_plot):
    """
    Generates depth temperature profiles at different points in time.
    Supports both laser modes (pulsed and continuous).
    
    Args:
        U_data (numpy.ndarray): Temperature data.
        x_plot (list): X-axis coordinates for plotting.
        y_plot (list): Y-axis coordinates for plotting.
        z_plot (list): Z-axis coordinates for plotting (depth).
        t_plot (list): Time points for plotting.
    
    Returns:
        None: Displays and saves the plot.
    """
    U_physical = convert_to_physical_temperature(U_data)
    z_phys = np.array(z_plot) * config.CHARACTERISTIC_LENGTH * 1e6  # мкм
    
    center_x = len(x_plot) // 2
    center_y = len(y_plot) // 2
    
    if config.LASER_MODE == "pulsed":
        # Для импульсного режима - моменты после каждого импульса
        key_frames = []
        frame_labels = []
        
        for i in range(min(config.NUM_PULSES, 8)):  # Максимум 8 кривых
            # Момент в середине импульса
            frame_idx = int((i + 0.5) * len(t_plot) / min(config.NUM_PULSES, 8))
            frame_idx = min(frame_idx, len(t_plot) - 1)
            key_frames.append(frame_idx)
            frame_labels.append(f'Во время {i+1} импульса')
        
        title_suffix = f'во время импульсов'
    else:
        # Для непрерывного режима - равномерные моменты времени
        key_frames = np.linspace(0, len(t_plot)-1, min(8, len(t_plot)), dtype=int)
        frame_labels = [f'Момент {i+1}' for i in range(len(key_frames))]
        title_suffix = 'в разные моменты времени'
    
    plt.figure(figsize=(12, 9))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(key_frames)))
    
    for i, (frame_idx, label) in enumerate(zip(key_frames, frame_labels)):
        temp_profile = U_physical[center_x, center_y, :, frame_idx]
        t_phys = t_plot[frame_idx] * config.CHARACTERISTIC_TIME * 1e6
        
        if config.LASER_MODE == "pulsed":
            pulse_num = int(t_plot[frame_idx] // config.LASER_PULSE_PERIOD_NORM) + 1
            label = f'Импульс {pulse_num} ({t_phys:.0f} мкс)'
        else:
            label = f'{t_phys:.0f} мкс'
        
        plt.plot(temp_profile, z_phys, color=colors[i], linewidth=2.5, label=label)
    
    # Добавляем изотерму 1900 K
    plt.axvline(x=1900, color='lime', linestyle='-', linewidth=2.5, alpha=0.7,
                label=f'Изотерма 1900 K')
    
    plt.xlabel('Температура (K)', fontsize=12)
    plt.ylabel('Глубина (мкм)', fontsize=12)
    
    mode_text = "Импульсный" if config.LASER_MODE == "pulsed" else "Непрерывный"
    power_text = f"{config.LASER_AVG_POWER} Вт (ср.)" if config.LASER_MODE == "pulsed" else f"{config.LASER_CONTINUOUS_POWER} Вт"
    
    title = f'Распределение температуры по глубине {title_suffix}\n'
    title += f'Режим: {mode_text}, Мощность: {power_text}'
    plt.title(title, fontsize=13)
    
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.gca().invert_yaxis()  # Глубина увеличивается вниз
    
    # Добавим горизонтальные линии на целевых глубинах
    target_depths = [10, 20, 30]
    for depth in target_depths:
        plt.axhline(y=depth, color='gray', linestyle=':', alpha=0.5, linewidth=1)
        plt.text(config.INITIAL_TEMPERATURE + 10, depth, f'{depth} мкм', 
                fontsize=9, color='gray', va='center')
    
    # Добавляем информацию о достижении изотермы
    max_temp_all = np.max(U_physical[center_x, center_y, :, :])
    if max_temp_all >= 1900:
        # Находим максимальную глубину достижения изотермы
        max_depth_1900 = 0
        for frame_idx in key_frames:
            temp_profile = U_physical[center_x, center_y, :, frame_idx]
            for i in range(len(temp_profile)):
                if temp_profile[i] >= 1900 and z_phys[i] > max_depth_1900:
                    max_depth_1900 = z_phys[i]
        
        plt.text(1900 + 50, max_depth_1900, f'Макс. глубина: {max_depth_1900:.1f} мкм',
                fontsize=10, color='lime', va='center',
                bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.5))
    
    plt.tight_layout()
    
    mode = config.LASER_MODE
    plt.savefig(f'animations/depth_temperature_profiles_{mode}.png', dpi=150, bbox_inches='tight')
    plt.show()

def plot_comparison_pulse_vs_continuous(temp_pulsed, temp_continuous, t_plot, 
                                        pulsed_params=None, continuous_params=None):
    """
    Generates a plot comparing temperature profiles for pulsed and continuous laser heating modes.
    
    The plot visualizes the temperature evolution over time for both modes, 
    highlights key parameters like maximum temperatures and the time at which they occur,
    and indicates whether the 1900 K isotherm is reached. 
    It also provides a detailed comparison of the two modes, including temperature differences and efficiency assessments.
    
    Args:
        temp_pulsed (array): Temperature data for the pulsed mode.
        temp_continuous (array): Temperature data for the continuous mode.
        t_plot (array): Time axis data (dimensionless).
        pulsed_params (dict, optional): Parameters for the pulsed mode. Defaults to {"num_pulses": 8, "avg_power": 10.0, "rep_rate": 8000.0}.
        continuous_params (dict, optional): Parameters for the continuous mode. Defaults to {"power": 5.0}.
    
    Returns:
        None: Displays and saves the generated plot as 'comparison_pulse_vs_continuous.png'. Prints a message if data for both modes is not provided.
    """
    if temp_pulsed is None or temp_continuous is None:
        print("Для сравнения нужны данные обоих режимов!")
        return
    
    t_phys = t_plot * config.CHARACTERISTIC_TIME * 1e6  # мкс
    
    plt.figure(figsize=(15, 8))
    
    # Графики температуры
    line_pulsed, = plt.plot(t_phys, temp_pulsed, 'r-', linewidth=3, alpha=0.8, label='Импульсный режим')
    line_continuous, = plt.plot(t_phys, temp_continuous, 'b-', linewidth=3, alpha=0.8, label='Непрерывный режим')
    
    # Начальная температура
    plt.axhline(y=config.INITIAL_TEMPERATURE, color='gray', linestyle='--', linewidth=2,
                alpha=0.5, label=f'Начальная ({config.INITIAL_TEMPERATURE} K)')
    
    # Изотерма 1900 K
    plt.axhline(y=1900, color='lime', linestyle='-', linewidth=2.5, alpha=0.8,
                label=f'Изотерма 1900 K')
    
    # Заполнение между кривыми для наглядности
    plt.fill_between(t_phys, temp_pulsed, temp_continuous, 
                     where=temp_pulsed >= temp_continuous, 
                     color='red', alpha=0.15, label='Импульсный > Непрерывный')
    plt.fill_between(t_phys, temp_pulsed, temp_continuous, 
                     where=temp_pulsed < temp_continuous, 
                     color='blue', alpha=0.15, label='Импульсный < Непрерывный')
    
    # Максимальные значения
    max_pulsed = np.max(temp_pulsed)
    max_continuous = np.max(temp_continuous)
    max_time_pulsed = t_phys[np.argmax(temp_pulsed)]
    max_time_continuous = t_phys[np.argmax(temp_continuous)]
    
    # Проверяем достижение изотермы 1900 K
    isotherm_pulsed = np.any(temp_pulsed >= 1900)
    isotherm_continuous = np.any(temp_continuous >= 1900)
    
    # Точки максимумов
    plt.scatter([max_time_pulsed], [max_pulsed], color='red', s=150, zorder=5,
                edgecolor='black', linewidth=2, label=f'Макс. импульсный: {max_pulsed:.1f} K')
    plt.scatter([max_time_continuous], [max_continuous], color='blue', s=150, zorder=5,
                edgecolor='black', linewidth=2, label=f'Макс. непрерывный: {max_continuous:.1f} K')
    
    # Вертикальные линии в точках максимумов
    plt.axvline(x=max_time_pulsed, color='red', linestyle=':', alpha=0.7, linewidth=1.5)
    plt.axvline(x=max_time_continuous, color='blue', linestyle=':', alpha=0.7, linewidth=1.5)
    
    # Параметры для информационного блока
    if pulsed_params is None:
        pulsed_params = {"num_pulses": 8, "avg_power": 10.0, "rep_rate": 8000.0}
    if continuous_params is None:
        continuous_params = {"power": 5.0}
    
    # Информация в текстовом блоке
    info_text = "СРАВНЕНИЕ РЕЖИМОВ ЛАЗЕРНОГО НАГРЕВА\n"
    info_text += "=" * 40 + "\n\n"
    
    info_text += "ИМПУЛЬСНЫЙ РЕЖИМ:\n"
    info_text += f"• Импульсов: {pulsed_params.get('num_pulses', 8)}\n"
    info_text += f"• Мощность: {pulsed_params.get('avg_power', 10.0)} Вт (ср.)\n"
    info_text += f"• Частота: {pulsed_params.get('rep_rate', 8000.0):.0f} Гц\n"
    info_text += f"• Макс. темп.: {max_pulsed:.1f} K\n"
    info_text += f"• Время макс.: {max_time_pulsed:.0f} мкс\n"
    info_text += f"• Изотерма 1900 K: {'ДА' if isotherm_pulsed else 'НЕТ'}\n\n"
    
    info_text += "НЕПРЕРЫВНЫЙ РЕЖИМ:\n"
    info_text += f"• Мощность: {continuous_params.get('power', 5.0)} Вт\n"
    info_text += f"• Макс. темп.: {max_continuous:.1f} K\n"
    info_text += f"• Время макс.: {max_time_continuous:.0f} мкс\n"
    info_text += f"• Изотерма 1900 K: {'ДА' if isotherm_continuous else 'НЕТ'}\n\n"
    
    info_text += "СРАВНЕНИЕ:\n"
    info_text += f"• Разница: {abs(max_pulsed - max_continuous):.1f} K\n"
    
    if max_pulsed > max_continuous:
        percent_diff = (max_pulsed/max_continuous - 1) * 100
        info_text += f"• Импульсный горячее на {percent_diff:.1f}%\n"
        info_text += "• Более эффективный нагрев\n"
    else:
        percent_diff = (max_continuous/max_pulsed - 1) * 100
        info_text += f"• Непрерывный горячее на {percent_diff:.1f}%\n"
        info_text += "• Более стабильный нагрев\n"
    
    info_text += f"• Начальная темп.: {config.INITIAL_TEMPERATURE} K"
    
    plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top', linespacing=1.5,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9))
    
    plt.xlabel('Время (мкс)', fontsize=12)
    plt.ylabel('Температура (K)', fontsize=12)
    plt.title('Сравнение температурных профилей: Импульсный vs Непрерывный режим', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Вторая легенда для заливки
    from matplotlib.patches import Patch
    legend_elements = [
        line_pulsed,
        line_continuous,
        Patch(facecolor='red', alpha=0.15, label='Импульсный > Непрерывный'),
        Patch(facecolor='blue', alpha=0.15, label='Импульсный < Непрерывный')
    ]
    
    plt.legend(handles=legend_elements, loc='lower right', fontsize=10)
    plt.tight_layout()
    plt.savefig('animations/comparison_pulse_vs_continuous.png', dpi=150, bbox_inches='tight')
    plt.show()

def plot_temperature_distribution_at_time(U_data, x_plot, y_plot, z_plot, t_plot, time_idx=-1):
    """
    Визуализирует распределение температуры в трехмерном пространстве в заданный момент времени, предоставляя детальный анализ температурного поля, включая поверхностные температуры, изотермы, профили по глубине и горизонтальные срезы.
    
    Args:
        U_data (numpy.ndarray): Данные о температуре.
        x_plot (numpy.ndarray): Координаты x для построения графика.
        y_plot (numpy.ndarray): Координаты y для построения графика.
        z_plot (numpy.ndarray): Координаты z для построения графика.
        t_plot (numpy.ndarray): Массив временных точек.
        time_idx (int, optional): Индекс времени для визуализации. По умолчанию -1 (последний момент времени).
    
    Returns:
        None:  Метод отображает графики, иллюстрирующие распределение температуры.
    """
    U_physical = convert_to_physical_temperature(U_data)
    x_phys = np.array(x_plot) * config.CHARACTERISTIC_LENGTH * 1e6
    y_phys = np.array(y_plot) * config.CHARACTERISTIC_LENGTH * 1e6
    z_phys = np.array(z_plot) * config.CHARACTERISTIC_LENGTH * 1e6
    
    t_phys = t_plot[time_idx] * config.CHARACTERISTIC_TIME * 1e6
    
    fig = plt.figure(figsize=(20, 12))
    
    # 1. 3D поверхность температуры на поверхности
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    X, Y = np.meshgrid(x_phys, y_phys)
    Z_surface = U_physical[:, :, -1, time_idx].T  # Поверхность (z = max)
    
    surf = ax1.plot_surface(X, Y, Z_surface, cmap='hot', alpha=0.85, 
                           rstride=1, cstride=1, linewidth=0.25, antialiased=True)
    ax1.set_xlabel('x (мкм)', fontsize=11, labelpad=10)
    ax1.set_ylabel('y (мкм)', fontsize=11, labelpad=10)
    ax1.set_zlabel('Температура (K)', fontsize=11, labelpad=10)
    ax1.set_title(f'Температура на поверхности\nВремя: {t_phys:.1f} мкс', fontsize=12)
    fig.colorbar(surf, ax=ax1, shrink=0.6, pad=0.1, label='Температура (K)')
    ax1.view_init(elev=30, azim=45)
    
    # 2. Изотермы на поверхности с выделением зоны >1900 K
    ax2 = fig.add_subplot(2, 3, 2)
    
    # Основной контурный график
    contour = ax2.contourf(X, Y, Z_surface, levels=20, cmap='hot')
    
    # Контур изотермы 1900 K
    if np.max(Z_surface) >= 1900:
        cs = ax2.contour(X, Y, Z_surface, levels=[1900], colors='lime', 
                        linewidths=3, alpha=0.9, linestyles='-')
        
        # Закрашиваем область внутри изотермы
        paths = cs.get_paths()
        for path in paths:
            vertices = path.vertices
            if len(vertices) > 2:
                polygon = Polygon(vertices, closed=True, fill=True,
                                 facecolor='lime', alpha=0.3, 
                                 edgecolor='lime', linewidth=2)
                ax2.add_patch(polygon)
    
    ax2.contour(X, Y, Z_surface, levels=10, colors='white', linewidths=0.5, alpha=0.5)
    ax2.set_xlabel('x (мкм)', fontsize=11)
    ax2.set_ylabel('y (мкм)', fontsize=11)
    ax2.set_title('Изотермы на поверхности (1900 K выделена)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Добавим круг радиуса пучка
    beam_radius_um = config.LASER_BEAM_RADIUS * 1e6
    circle = Circle((0, 0), beam_radius_um, fill=False, color='cyan', 
                       linestyle='-', linewidth=2, alpha=0.7)
    ax2.add_patch(circle)
    ax2.text(0, -beam_radius_um*1.1, f'Радиус пучка: {beam_radius_um:.0f} мкм', 
            ha='center', color='cyan', fontsize=10)
    
    fig.colorbar(contour, ax=ax2, shrink=0.9, label='Температура (K)')
    ax2.set_aspect('equal')
    
    # 3. Распределение по глубине в центре
    ax3 = fig.add_subplot(2, 3, 3)
    center_x = len(x_plot) // 2
    center_y = len(y_plot) // 2
    temp_profile = U_physical[center_x, center_y, :, time_idx]
    
    ax3.plot(temp_profile, z_phys, 'b-', linewidth=3)
    
    # Добавляем изотерму 1900 K
    ax3.axvline(x=1900, color='lime', linestyle='-', linewidth=2.5, alpha=0.7,
                label=f'Изотерма 1900 K')
    
    ax3.set_xlabel('Температура (K)', fontsize=11)
    ax3.set_ylabel('Глубина z (мкм)', fontsize=11)
    ax3.set_title('Распределение по глубине (центр пучка)', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.invert_yaxis()  # Глубина увеличивается вниз
    
    # Отметим температуры на целевых глубинах
    target_depths = [10, 20, 30]
    colors = ['cyan', 'lime', 'yellow']
    
    for depth, color in zip(target_depths, colors):
        depth_idx = np.argmin(np.abs(z_phys - depth))
        temp = temp_profile[depth_idx]
        ax3.plot(temp, depth, 'o', color=color, markersize=10, 
                markeredgecolor='black', linewidth=1.5)
        ax3.text(temp + 5, depth, f'{temp:.0f} K', fontsize=10, va='center',
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))
    
    ax3.legend(fontsize=10)
    
    # 4. Горизонтальный срез на глубине 20 мкм
    ax4 = fig.add_subplot(2, 3, 4)
    depth_idx = np.argmin(np.abs(z_phys - 20))
    actual_depth = z_phys[depth_idx]
    depth_slice = U_physical[:, :, depth_idx, time_idx].T
    
    im4 = ax4.imshow(depth_slice, extent=[x_phys[0], x_phys[-1], y_phys[0], y_phys[-1]], 
                    origin='lower', aspect='auto', cmap='hot')
    ax4.set_xlabel('x (мкм)', fontsize=11)
    ax4.set_ylabel('y (мкм)', fontsize=11)
    ax4.set_title(f'Срез на глубине {actual_depth:.0f} мкм', fontsize=12)
    ax4.grid(True, alpha=0.3)
    
    # Перекрестие в центре
    ax4.axhline(y=0, color='white', linestyle='--', alpha=0.5, linewidth=1)
    ax4.axvline(x=0, color='white', linestyle='--', alpha=0.5, linewidth=1)
    
    fig.colorbar(im4, ax=ax4, shrink=0.9, label='Температура (K)')
    
    # 5. Профиль вдоль оси X на поверхности
    ax5 = fig.add_subplot(2, 3, 5)
    y_center_idx = len(y_plot) // 2
    x_profile = U_physical[:, y_center_idx, -1, time_idx]
    
    ax5.plot(x_phys, x_profile, 'r-', linewidth=3, label='Поверхность (y=0)')
    ax5.fill_between(x_phys, config.INITIAL_TEMPERATURE, x_profile, alpha=0.2, color='red')
    
    # Добавляем изотерму 1900 K
    ax5.axhline(y=1900, color='lime', linestyle='-', linewidth=2.5, alpha=0.8,
                label=f'Изотерма 1900 K')
    
    ax5.set_xlabel('x (мкм)', fontsize=11)
    ax5.set_ylabel('Температура (K)', fontsize=11)
    ax5.set_title('Профиль температуры вдоль оси X', fontsize=12)
    ax5.grid(True, alpha=0.3)
    
    # Отметим радиус пучка
    ax5.axvline(x=beam_radius_um, color='cyan', linestyle='--', alpha=0.7, linewidth=2)
    ax5.axvline(x=-beam_radius_um, color='cyan', linestyle='--', alpha=0.7, linewidth=2)
    ax5.text(beam_radius_um, config.INITIAL_TEMPERATURE + 10, f'Радиус пучка', 
            fontsize=10, color='cyan', ha='center')
    
    ax5.legend(fontsize=10)
    
    # 6. Информационная панель с акцентом на изотерму 1900 K
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    
    max_temp = np.max(U_physical[:, :, :, time_idx])
    min_temp = np.min(U_physical[:, :, :, time_idx])
    avg_temp = np.mean(U_physical[:, :, :, time_idx])
    
    # Рассчитываем параметры изотермы 1900 K
    isotherm_reached = max_temp >= 1900
    width_1900 = None
    depth_1900 = None
    
    if isotherm_reached:
        # Ширина на поверхности
        surface_temp = U_physical[:, center_y, -1, time_idx]
        left_idx = None
        right_idx = None
        
        for i in range(center_x, len(x_phys)):
            if surface_temp[i] >= 1900 and right_idx is None:
                right_idx = i
            if surface_temp[center_x - (i-center_x)] >= 1900 and left_idx is None:
                left_idx = center_x - (i-center_x)
        
        if left_idx is not None and right_idx is not None:
            width_1900 = x_phys[right_idx] - x_phys[left_idx]
        
        # Глубина проникновения 1900 K
        center_temp_profile = U_physical[center_x, center_y, :, time_idx]
        for i in range(len(z_phys)):
            if center_temp_profile[i] >= 1900:
                depth_1900 = z_phys[i]
                break
    
    info_text = f"АНАЛИЗ ТЕМПЕРАТУРНОГО ПОЛЯ\n"
    info_text += "=" * 40 + "\n\n"
    info_text += f"ВРЕМЯ: {t_phys:.1f} мкс\n\n"
    
    info_text += f"ТЕМПЕРАТУРНЫЕ ХАРАКТЕРИСТИКИ:\n"
    info_text += f"• Максимальная: {max_temp:.1f} K\n"
    info_text += f"• Минимальная: {min_temp:.1f} K\n"
    info_text += f"• Средняя: {avg_temp:.1f} K\n"
    info_text += f"• Перегрев: {max_temp - config.INITIAL_TEMPERATURE:.1f} K\n\n"
    
    info_text += f"ИЗОТЕРМА 1900 K:\n"
    info_text += f"• Достигнута: {'ДА' if isotherm_reached else 'НЕТ'}\n"
    if width_1900 is not None:
        info_text += f"• Ширина зоны: {width_1900:.1f} мкм\n"
    if depth_1900 is not None:
        info_text += f"• Глубина проникновения: {depth_1900:.1f} мкм\n"
    
    if config.LASER_MODE == "pulsed":
        pulse_number = min(int(t_plot[time_idx] // config.LASER_PULSE_PERIOD_NORM) + 1, config.NUM_PULSES)
        info_text += f"\nРЕЖИМ: ИМПУЛЬСНЫЙ\n"
        info_text += f"• Импульс: {pulse_number}/{config.NUM_PULSES}\n"
        info_text += f"• Мощность: {config.LASER_AVG_POWER} Вт (ср.)\n"
        info_text += f"• Частота: {config.LASER_REP_RATE:.0f} Гц\n"
    else:
        info_text += f"\nРЕЖИМ: НЕПРЕРЫВНЫЙ\n"
        info_text += f"• Мощность: {config.LASER_CONTINUOUS_POWER} Вт\n"
        info_text += f"• Интенсивность: {config.LASER_PEAK_INTENSITY/1e6:.2f} МВт/м²\n"
    
    info_text += f"\nПАРАМЕТРЫ ПУЧКА:\n"
    info_text += f"• Радиус: {config.LASER_BEAM_RADIUS*1e6:.0f} мкм\n"
    info_text += f"• Длина волны: {config.LASER_WAVELENGTH*1e6:.1f} мкм\n\n"
    
    info_text += f"ТЕМПЕРАТУРА НА ГЛУБИНАХ:\n"
    for depth, color in zip(target_depths, colors):
        depth_idx = np.argmin(np.abs(z_phys - depth))
        temp = U_physical[center_x, center_y, depth_idx, time_idx]
        info_text += f"• {depth} мкм: {temp:.1f} K\n"
    
    ax6.text(0.1, 0.95, info_text, fontsize=11, va='top', linespacing=1.6,
             bbox=dict(boxstyle="round,pad=0.6", facecolor="lightblue", alpha=0.9))
    
    mode_text = "Импульсный" if config.LASER_MODE == "pulsed" else "Непрерывный"
    power_text = f"{config.LASER_AVG_POWER} Вт (ср.)" if config.LASER_MODE == "pulsed" else f"{config.LASER_CONTINUOUS_POWER} Вт"
    
    plt.suptitle(f'Детальный анализ температурного поля\nРежим: {mode_text}, Мощность: {power_text}, Время: {t_phys:.1f} мкс', 
                fontsize=15, y=0.98)
    
    plt.tight_layout()
    
    mode = config.LASER_MODE
    plt.savefig(f'animations/temperature_field_analysis_{mode}_t{t_phys:.0f}us.png', 
               dpi=150, bbox_inches='tight')
    plt.show()

def plot_laser_intensity_3d():
    """
    Визуализирует интенсивность лазерного пучка в 3D и 2D, отображая его профиль и ключевые параметры.
    
    Args:
        None
    
    Returns:
        None
    """
    x = np.linspace(-1, 1, 100) * config.CHARACTERISTIC_LENGTH * 1e6
    y = np.linspace(-1, 1, 100) * config.CHARACTERISTIC_LENGTH * 1e6
    X, Y = np.meshgrid(x, y)
    
    # Вычисляем интенсивность в относительных единицах
    intensity = np.exp(-(X**2 + Y**2) / (config.LASER_BEAM_RADIUS*1e6)**2)
    
    fig = plt.figure(figsize=(16, 8))
    
    # 3D поверхность
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    surf = ax1.plot_surface(X, Y, intensity, cmap='hot', alpha=0.85, 
                           rstride=2, cstride=2, linewidth=0.3, antialiased=True,
                           edgecolor='black')
    ax1.set_xlabel('x (мкм)', fontsize=12, labelpad=10)
    ax1.set_ylabel('y (мкм)', fontsize=12, labelpad=10)
    ax1.set_zlabel('Относительная интенсивность', fontsize=12, labelpad=10)
    ax1.set_title('3D профиль лазерного пучка', fontsize=13)
    fig.colorbar(surf, ax=ax1, shrink=0.6, pad=0.1, label='Интенсивность')
    ax1.view_init(elev=35, azim=45)
    
    # 2D контурный график
    ax2 = fig.add_subplot(1, 2, 2)
    contour = ax2.contourf(X, Y, intensity, levels=30, cmap='hot')
    ax2.contour(X, Y, intensity, levels=15, colors='white', linewidths=0.5, alpha=0.5)
    
    # Добавим круги на различных радиусах
    radius = config.LASER_BEAM_RADIUS * 1e6
    
    # Полный радиус пучка
    circle_full = Circle((0, 0), radius, fill=False, color='cyan', 
                            linestyle='-', linewidth=3, alpha=0.8, 
                            label=f'Радиус пучка: {radius:.0f} мкм')
    ax2.add_patch(circle_full)
    
    # Радиус на уровне 1/e² (13.5% от максимума)
    radius_1e2 = radius / np.sqrt(2)
    circle_1e2 = Circle((0, 0), radius_1e2, fill=False, color='lime', 
                           linestyle='--', linewidth=2, alpha=0.7, 
                           label=f'1/e² радиус: {radius_1e2:.0f} мкм')
    ax2.add_patch(circle_1e2)
    
    # Радиус на уровне 1/e (36.8% от максимума)
    radius_1e = radius
    circle_1e = Circle((0, 0), radius_1e, fill=False, color='yellow', 
                          linestyle=':', linewidth=2, alpha=0.6, 
                          label=f'1/e радиус: {radius_1e:.0f} мкм')
    ax2.add_patch(circle_1e)
    
    ax2.set_xlabel('x (мкм)', fontsize=12)
    ax2.set_ylabel('y (мкм)', fontsize=12)
    ax2.set_title('2D распределение интенсивности', fontsize=13)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.set_aspect('equal')
    fig.colorbar(contour, ax=ax2, shrink=0.9, label='Интенсивность')
    
    mode_text = "Импульсный" if config.LASER_MODE == "pulsed" else "Непрерывный"
    power_text = f"{config.LASER_AVG_POWER} Вт (ср.)" if config.LASER_MODE == "pulsed" else f"{config.LASER_CONTINUOUS_POWER} Вт"
    
    plt.suptitle(f'Лазерный пучок: {mode_text} режим\nМощность: {power_text}, λ={config.LASER_WAVELENGTH*1e6:.1f} мкм, Радиус: {radius:.0f} мкм',
                fontsize=14, y=0.95)
    
    plt.tight_layout()
    plt.savefig('animations/laser_intensity_3d_profile.png', dpi=150, bbox_inches='tight')
    plt.show()

def plot_heating_dynamics_comparison(U_data_pulsed, U_data_continuous, 
                                    x_plot, y_plot, z_plot, t_plot):
    """
    Сравнивает динамику нагрева в импульсном и непрерывном режимах, визуализируя изменения температуры в центре образца, градиент температуры, объем нагретого материала и интегральную характеристику перегрева.
    
    Args:
        U_data_pulsed: Температурное поле для импульсного режима.
        U_data_continuous: Температурное поле для непрерывного режима.
        x_plot: Координаты x для построения графиков.
        y_plot: Координаты y для построения графиков.
        z_plot: Координаты z для построения графиков.
        t_plot: Временные точки для построения графиков.
    
    Returns:
        None. Отображает графики сравнения динамики нагрева.
    """
    U_phys_pulsed = convert_to_physical_temperature(U_data_pulsed)
    U_phys_continuous = convert_to_physical_temperature(U_data_continuous)
    t_phys = t_plot * config.CHARACTERISTIC_TIME * 1e6
    
    center_x = len(x_plot) // 2
    center_y = len(y_plot) // 2
    surface_z = len(z_plot) - 1
    
    # Температура в центре
    temp_center_pulsed = U_phys_pulsed[center_x, center_y, surface_z, :]
    temp_center_continuous = U_phys_continuous[center_x, center_y, surface_z, :]
    
    fig = plt.figure(figsize=(18, 12))
    
    # 1. Сравнение температур в центре
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(t_phys, temp_center_pulsed, 'r-', linewidth=3, label='Импульсный режим', alpha=0.8)
    ax1.plot(t_phys, temp_center_continuous, 'b-', linewidth=3, label='Непрерывный режим', alpha=0.8)
    ax1.axhline(y=config.INITIAL_TEMPERATURE, color='gray', linestyle='--', 
                linewidth=2, alpha=0.5, label=f'Начальная ({config.INITIAL_TEMPERATURE} K)')
    
    # Изотерма 1900 K
    ax1.axhline(y=1900, color='lime', linestyle='-', linewidth=2.5, alpha=0.8,
                label=f'Изотерма 1900 K')
    
    ax1.set_xlabel('Время (мкс)', fontsize=12)
    ax1.set_ylabel('Температура (K)', fontsize=12)
    ax1.set_title('Сравнение температуры в центре пучка', fontsize=13)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    # 2. Градиент температуры
    ax2 = fig.add_subplot(2, 2, 2)
    
    # Рассчитаем градиент температуры (разница между поверхностью и глубиной 30 мкм)
    depth_idx = np.argmin(np.abs(np.array(z_plot) * config.CHARACTERISTIC_LENGTH * 1e6 - 30))
    
    temp_surface_pulsed = U_phys_pulsed[center_x, center_y, surface_z, :]
    temp_depth_pulsed = U_phys_pulsed[center_x, center_y, depth_idx, :]
    gradient_pulsed = temp_surface_pulsed - temp_depth_pulsed
    
    temp_surface_continuous = U_phys_continuous[center_x, center_y, surface_z, :]
    temp_depth_continuous = U_phys_continuous[center_x, center_y, depth_idx, :]
    gradient_continuous = temp_surface_continuous - temp_depth_continuous
    
    ax2.plot(t_phys, gradient_pulsed, 'r-', linewidth=2.5, label='Импульсный режим', alpha=0.8)
    ax2.plot(t_phys, gradient_continuous, 'b-', linewidth=2.5, label='Непрерывный режим', alpha=0.8)
    
    ax2.set_xlabel('Время (мкс)', fontsize=12)
    ax2.set_ylabel('Градиент температуры (K)', fontsize=12)
    ax2.set_title('Градиент температуры (поверхность - глубина 30 мкм)', fontsize=13)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    
    # 3. Объем нагретого материала выше 1900 K
    ax3 = fig.add_subplot(2, 2, 3)
    
    # Определим объем материала с температурой выше 1900 K
    threshold_temp = 1900  # Изотерма 1900 K
    
    heated_volume_pulsed = []
    heated_volume_continuous = []
    
    for t_idx in range(len(t_plot)):
        # Для импульсного режима
        above_threshold_pulsed = np.sum(U_phys_pulsed[:, :, :, t_idx] > threshold_temp)
        heated_volume_pulsed.append(above_threshold_pulsed)
        
        # Для непрерывного режима
        above_threshold_continuous = np.sum(U_phys_continuous[:, :, :, t_idx] > threshold_temp)
        heated_volume_continuous.append(above_threshold_continuous)
    
    heated_volume_pulsed = np.array(heated_volume_pulsed)
    heated_volume_continuous = np.array(heated_volume_continuous)
    
    # Нормализуем к максимальному значению
    max_volume = max(np.max(heated_volume_pulsed), np.max(heated_volume_continuous))
    if max_volume > 0:
        heated_volume_pulsed = heated_volume_pulsed / max_volume * 100
        heated_volume_continuous = heated_volume_continuous / max_volume * 100
    
    ax3.plot(t_phys, heated_volume_pulsed, 'r-', linewidth=2.5, label='Импульсный режим', alpha=0.8)
    ax3.plot(t_phys, heated_volume_continuous, 'b-', linewidth=2.5, label='Непрерывный режим', alpha=0.8)
    
    ax3.set_xlabel('Время (мкс)', fontsize=12)
    ax3.set_ylabel('Относительный объем нагретого материала (%)', fontsize=12)
    ax3.set_title(f'Объем материала с T > 1900K (относительно максимума)', fontsize=13)
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=10)
    ax3.set_ylim(0, 105)
    
    # 4. Интегральная характеристика нагрева выше 1900 K
    ax4 = fig.add_subplot(2, 2, 4)
    
    # Интеграл от превышения температуры над 1900 K (если достигнута)
    integral_pulsed = np.zeros_like(t_phys)
    integral_continuous = np.zeros_like(t_phys)
    
    for t_idx in range(len(t_plot)):
        if temp_center_pulsed[t_idx] > 1900:
            integral_pulsed[t_idx] = np.sum(temp_center_pulsed[:t_idx+1] - 1900)
        if temp_center_continuous[t_idx] > 1900:
            integral_continuous[t_idx] = np.sum(temp_center_continuous[:t_idx+1] - 1900)
    
    ax4.plot(t_phys, integral_pulsed, 'r-', linewidth=2.5, label='Импульсный режим', alpha=0.8)
    ax4.plot(t_phys, integral_continuous, 'b-', linewidth=2.5, label='Непрерывный режим', alpha=0.8)
    
    ax4.set_xlabel('Время (мкс)', fontsize=12)
    ax4.set_ylabel('Интеграл перегрева (K·мкс)', fontsize=12)
    ax4.set_title('Накопленный перегрев выше 1900K (интегральная характеристика)', fontsize=13)
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=10)
    
    plt.suptitle('Сравнительный анализ динамики лазерного нагрева: Импульсный vs Непрерывный режимы', 
                fontsize=15, y=0.98)
    
    plt.tight_layout()
    plt.savefig('animations/heating_dynamics_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

def plot_pulse_train_visualization():
    """
    Визуализация последовательности импульсов для импульсного режима.
    
    Метод генерирует и отображает график последовательности лазерных импульсов,
    позволяя визуально оценить параметры импульсного режима, такие как период,
    длительность и амплитуда импульсов.
    
    Args:
        None
    
    Returns:
        None
    """
    if config.LASER_MODE != "pulsed":
        print("Эта функция предназначена только для импульсного режима!")
        return
    
    # Создаем последовательность импульсов
    total_time = config.NUM_PULSES * config.LASER_PULSE_PERIOD_NORM
    t_test = np.linspace(0, total_time, 5000)
    
    source_values = np.zeros_like(t_test)
    pulse_peaks = []
    
    for i, t_val in enumerate(t_test):
        t_mod = t_val % config.LASER_PULSE_PERIOD_NORM
        pulse_value = config.LASER_AMPLITUDE * np.exp(
            -(t_mod - config.LASER_PULSE_PERIOD_NORM/2)**2 / 
            (2 * config.LASER_PULSE_SIGMA_NORM**2)
        )
        source_values[i] = pulse_value
        
        # Запоминаем пики импульсов
        if t_mod < 0.1 * config.LASER_PULSE_PERIOD_NORM and i > 0:
            pulse_peaks.append(t_val)
    
    # Конвертация в физическое время
    t_phys = t_test * config.CHARACTERISTIC_TIME * 1e6
    
    plt.figure(figsize=(14, 8))
    
    # Основной график
    plt.plot(t_phys, source_values, 'r-', linewidth=2.5, alpha=0.8)
    
    # Заливка под кривой
    plt.fill_between(t_phys, 0, source_values, alpha=0.2, color='red')
    
    # Вертикальные линии для каждого импульса
    for i in range(config.NUM_PULSES + 1):
        impulse_time = i * config.LASER_PULSE_PERiod_NORM * config.CHARACTERISTIC_TIME * 1e6
        plt.axvline(x=impulse_time, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        
        if i < config.NUM_PULSES:
            # Номер импульса
            plt.text(impulse_time + 5, 0.9, f'{i+1}', fontsize=10, fontweight='bold', 
                    color='darkred', ha='left')
            
            # Область импульса
            impulse_end = impulse_time + config.LASER_PULSE_DURATION_NORM * config.CHARACTERISTIC_TIME * 1e6
            plt.axvspan(impulse_time, impulse_end, alpha=0.1, color='red')
    
    # Горизонтальная линия на уровне 1/e²
    level_1e2 = config.LASER_AMPLITUDE / np.e**2
    plt.axhline(y=level_1e2, color='orange', linestyle=':', alpha=0.7, linewidth=1.5,
               label=f'Уровень 1/e² ({level_1e2:.2f})')
    
    plt.xlabel('Время (мкс)', fontsize=12)
    plt.ylabel('Относительная интенсивность', fontsize=12)
    
    title = f'Последовательность лазерных импульсов\n'
    title += f'{config.NUM_PULSES} импульсов, период {config.LASER_PULSE_PERIOD*1e6:.1f} мкс, '
    title += f'длительность {config.LASER_PULSE_DURATION*1e6:.1f} мкс'
    plt.title(title, fontsize=13)
    
    plt.grid(True, alpha=0.3)
    plt.ylim(0, config.LASER_AMPLITUDE * 1.05)
    
    # Информационная панель
    info_text = f"ПАРАМЕТРЫ ПОСЛЕДОВАТЕЛЬНОСТИ ИМПУЛЬСОВ:\n"
    info_text += "=" * 40 + "\n\n"
    info_text += f"• Количество импульсов: {config.NUM_PULSES}\n"
    info_text += f"• Период: {config.LASER_PULSE_PERIOD*1e6:.1f} мкс\n"
    info_text += f"• Длительность импульса: {config.LASER_PULSE_DURATION*1e6:.1f} мкс\n"
    info_text += f"• Частота: {config.LASER_REP_RATE:.0f} Гц\n"
    info_text += f"• Скважность: {config.LASER_DUTY_CYCLE*100:.1f}%\n"
    info_text += f"• Пиковая мощность: {config.LASER_PEAK_POWER:.1f} Вт\n"
    info_text += f"• Средняя мощность: {config.LASER_AVG_POWER} Вт\n"
    info_text += f"• Общее время: {total_time*config.CHARACTERISTIC_TIME*1e6:.1f} мкс"
    
    plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top', linespacing=1.5,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9))
    
    plt.legend(loc='upper right', fontsize=10)
    plt.tight_layout()
    plt.savefig('animations/pulse_train_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()

def plot_isotherm_1900_analysis(U_data, x_plot, y_plot, z_plot, t_plot, time_idx=-1):
    """
    Generates a detailed analysis of the 1900 K isotherm, visualizing temperature distribution and key parameters over time.
    
    This function creates several plots to understand the thermal behavior of the material under laser irradiation, 
    including surface temperature contours, temperature profiles, 3D visualization of the heated zone, and the evolution 
    of zone characteristics over time. It provides insights into the extent and dynamics of the heated area.
    
    Args:
        U_data (numpy.ndarray): 4D array containing temperature data.
        x_plot (numpy.ndarray): 1D array representing x-coordinates.
        y_plot (numpy.ndarray): 1D array representing y-coordinates.
        z_plot (numpy.ndarray): 1D array representing z-coordinates.
        t_plot (numpy.ndarray): 1D array representing time points.
        time_idx (int, optional): Index of the time point to analyze. Defaults to -1 (last time point).
    
    Returns:
        None: Displays the generated plots and saves them as PNG images.
    """
    U_physical = convert_to_physical_temperature(U_data)
    x_phys = np.array(x_plot) * config.CHARACTERISTIC_LENGTH * 1e6
    y_phys = np.array(y_plot) * config.CHARACTERISTIC_LENGTH * 1e6
    z_phys = np.array(z_plot) * config.CHARACTERISTIC_LENGTH * 1e6
    
    t_phys = t_plot[time_idx] * config.CHARACTERISTIC_TIME * 1e6
    
    fig = plt.figure(figsize=(18, 12))
    
    # Центральные индексы
    center_x = len(x_plot) // 2
    center_y = len(y_plot) // 2
    
    # 1. Изотерма на поверхности
    ax1 = fig.add_subplot(2, 3, 1)
    X, Y = np.meshgrid(x_phys, y_phys)
    Z_surface = U_physical[:, :, -1, time_idx].T
    
    # Основной контурный график
    contour1 = ax1.contourf(X, Y, Z_surface, levels=20, cmap='hot')
    
    # Контур изотермы 1900 K
    if np.max(Z_surface) >= 1900:
        cs = ax1.contour(X, Y, Z_surface, levels=[1900], colors='lime', 
                        linewidths=3, alpha=0.9, linestyles='-')
        
        # Закрашиваем область внутри изотермы
        paths = cs.get_paths()
        for path in paths:
            vertices = path.vertices
            if len(vertices) > 2:
                polygon = Polygon(vertices, closed=True, fill=True,
                                 facecolor='lime', alpha=0.3, 
                                 edgecolor='lime', linewidth=2)
                ax1.add_patch(polygon)
    
    ax1.set_xlabel('x (мкм)', fontsize=11)
    ax1.set_ylabel('y (мкм)', fontsize=11)
    ax1.set_title(f'Изотерма 1900 K на поверхности\nВремя: {t_phys:.1f} мкс', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    fig.colorbar(contour1, ax=ax1, shrink=0.8, label='Температура (K)')
    
    # 2. Профиль вдоль X через центр
    ax2 = fig.add_subplot(2, 3, 2)
    x_profile = U_physical[:, center_y, -1, time_idx]
    
    ax2.plot(x_phys, x_profile, 'b-', linewidth=2.5)
    ax2.fill_between(x_phys, config.INITIAL_TEMPERATURE, x_profile, alpha=0.2, color='blue')
    
    # Изотерма 1900 K
    ax2.axhline(y=1900, color='lime', linestyle='-', linewidth=2.5, alpha=0.8,
                label='Изотерма 1900 K')
    
    # Находим ширину зоны >1900 K
    width_1900 = None
    left_idx = None
    right_idx = None
    
    for i in range(center_x, len(x_phys)):
        if x_profile[i] >= 1900 and right_idx is None:
            right_idx = i
        if x_profile[center_x - (i-center_x)] >= 1900 and left_idx is None:
            left_idx = center_x - (i-center_x)
    
    if left_idx is not None and right_idx is not None:
        width_1900 = x_phys[right_idx] - x_phys[left_idx]
        # Закрашиваем зону >1900 K
        ax2.fill_between(x_phys[left_idx:right_idx+1], 1900, x_profile[left_idx:right_idx+1], 
                        alpha=0.3, color='lime', label=f'Зона >1900 K ({width_1900:.1f} мкм)')
    
    ax2.set_xlabel('x (мкм)', fontsize=11)
    ax2.set_ylabel('Температура (K)', fontsize=11)
    ax2.set_title('Профиль температуры вдоль X (y=0)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    
    # 3. Профиль по глубине в центре
    ax3 = fig.add_subplot(2, 3, 3)
    depth_profile = U_physical[center_x, center_y, :, time_idx]
    
    ax3.plot(depth_profile, z_phys, 'r-', linewidth=2.5)
    
    # Изотерма 1900 K
    ax3.axvline(x=1900, color='lime', linestyle='-', linewidth=2.5, alpha=0.8,
                label='Изотерма 1900 K')
    
    # Находим глубину проникновения 1900 K
    depth_1900 = None
    for i in range(len(z_phys)):
        if depth_profile[i] >= 1900:
            depth_1900 = z_phys[i]
            # Закрашиваем зону >1900 K по глубине
            ax3.fill_betweenx(z_phys[:i+1], 1900, depth_profile[:i+1], 
                            alpha=0.3, color='lime', 
                            label=f'Глубина: {depth_1900:.1f} мкм')
            break
    
    ax3.set_xlabel('Температура (K)', fontsize=11)
    ax3.set_ylabel('Глубина z (мкм)', fontsize=11)
    ax3.set_title('Профиль температуры по глубине (центр)', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.invert_yaxis()
    ax3.legend(fontsize=10)
    
    # 4. 3D визуализация зоны >1900 K
    ax4 = fig.add_subplot(2, 3, 4, projection='3d')
    
    # Создаем маску для зоны >1900 K
    mask_1900 = U_physical[:, :, :, time_idx] > 1900
    
    if np.any(mask_1900):
        # Получаем координаты точек в зоне >1900 K
        x_idx, y_idx, z_idx = np.where(mask_1900)
        x_points = x_phys[x_idx]
        y_points = y_phys[y_idx]
        z_points = z_phys[z_idx]
        temp_points = U_physical[x_idx, y_idx, z_idx, time_idx]
        
        # Рисуем точки в 3D
        sc = ax4.scatter(x_points, y_points, -z_points, c=temp_points, 
                        cmap='hot', alpha=0.6, s=10, vmin=1900)
        
        # Подписываем количество точек
        ax4.text2D(0.05, 0.95, f'Точек >1900K: {len(x_points)}', 
                  transform=ax4.transAxes, fontsize=10, fontweight='bold',
                  bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    else:
        ax4.text2D(0.3, 0.5, "Изотерма 1900 K\nне достигнута", 
                  transform=ax4.transAxes, fontsize=12, fontweight='bold',
                  color='red', ha='center')
    
    ax4.set_xlabel('x (мкм)', fontsize=11, labelpad=10)
    ax4.set_ylabel('y (мкм)', fontsize=11, labelpad=10)
    ax4.set_zlabel('Глубина (мкм)', fontsize=11, labelpad=10)
    ax4.set_title('3D распределение зоны >1900 K', fontsize=12)
    ax4.view_init(elev=25, azim=45)
    
    if np.any(mask_1900):
        fig.colorbar(sc, ax=ax4, shrink=0.6, pad=0.1, label='Температура (K)')
    
    # 5. Эволюция размеров зоны >1900 K во времени
    ax5 = fig.add_subplot(2, 3, 5)
    
    widths = []
    depths = []
    volumes = []
    
    for t_idx in range(len(t_plot)):
        # Ширина на поверхности
        surface_temp = U_physical[:, center_y, -1, t_idx]
        left_idx = None
        right_idx = None
        
        for i in range(center_x, len(x_phys)):
            if surface_temp[i] >= 1900 and right_idx is None:
                right_idx = i
            if surface_temp[center_x - (i-center_x)] >= 1900 and left_idx is None:
                left_idx = center_x - (i-center_x)
        
        width = 0
        if left_idx is not None and right_idx is not None:
            width = x_phys[right_idx] - x_phys[left_idx]
        widths.append(width)
        
        # Глубина проникновения
        center_temp_profile = U_physical[center_x, center_y, :, t_idx]
        depth = 0
        for i in range(len(z_phys)):
            if center_temp_profile[i] >= 1900:
                depth = z_phys[i]
        depths.append(depth)
        
        # Объем (количество точек >1900 K)
        volume = np.sum(U_physical[:, :, :, t_idx] > 1900)
        volumes.append(volume)
    
    widths = np.array(widths)
    depths = np.array(depths)
    volumes = np.array(volumes)
    
    # Нормализуем объем
    if np.max(volumes) > 0:
        volumes = volumes / np.max(volumes) * 100
    
    ax5.plot(t_phys, widths, 'b-', linewidth=2, label='Ширина зоны (мкм)')
    ax5.plot(t_phys, depths, 'r-', linewidth=2, label='Глубина (мкм)')
    ax5.plot(t_phys, volumes, 'g-', linewidth=2, label='Отн. объем (%)')
    
    ax5.set_xlabel('Время (мкс)', fontsize=11)
    ax5.set_ylabel('Параметры зоны >1900 K', fontsize=11)
    ax5.set_title('Эволюция зоны >1900 K во времени', fontsize=12)
    ax5.grid(True, alpha=0.3)
    ax5.legend(fontsize=10)
    
    # 6. Информационная панель
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    
    max_temp = np.max(U_physical[:, :, :, time_idx])
    
    # Текущие параметры
    current_width = widths[time_idx] if time_idx < len(widths) else widths[-1]
    current_depth = depths[time_idx] if time_idx < len(depths) else depths[-1]
    current_volume = volumes[time_idx] if time_idx < len(volumes) else volumes[-1]
    
    # Максимальные параметры за все время
    max_width = np.max(widths)
    max_depth = np.max(depths)
    max_volume_idx = np.argmax(volumes)
    time_max_volume = t_phys[max_volume_idx]
    
    info_text = f"АНАЛИЗ ИЗОТЕРМЫ 1900 K\n"
    info_text += "=" * 30 + "\n\n"
    info_text += f"Текущее время: {t_phys:.1f} мкс\n\n"
    
    info_text += f"ТЕКУЩИЕ ПАРАМЕТРЫ:\n"
    info_text += f"• Макс. темп.: {max_temp:.1f} K\n"
    info_text += f"• Ширина зоны: {current_width:.1f} мкм\n"
    info_text += f"• Глубина: {current_depth:.1f} мкм\n"
    info_text += f"• Отн. объем: {current_volume:.1f}%\n\n"
    
    info_text += f"МАКСИМАЛЬНЫЕ ЗНАЧЕНИЯ:\n"
    info_text += f"• Макс. ширина: {max_width:.1f} мкм\n"
    info_text += f"• Макс. глубина: {max_depth:.1f} мкм\n"
    info_text += f"• Макс. объем: {np.max(volumes):.1f}%\n"
    info_text += f"• Время макс. объема: {time_max_volume:.1f} мкс\n\n"
    
    if config.LASER_MODE == "pulsed":
        pulse_number = min(int(t_plot[time_idx] // config.LASER_PULSE_PERIOD_NORM) + 1, config.NUM_PULSES)
        info_text += f"РЕЖИМ: ИМПУЛЬСНЫЙ\n"
        info_text += f"• Импульс: {pulse_number}/{config.NUM_PULSES}\n"
    else:
        info_text += f"РЕЖИМ: НЕПРЕРЫВНЫЙ\n"
    
    ax6.text(0.1, 0.95, info_text, fontsize=11, va='top', linespacing=1.5,
             bbox=dict(boxstyle="round,pad=0.6", facecolor="lightgreen", alpha=0.9))
    
    mode_text = "Импульсный" if config.LASER_MODE == "pulsed" else "Непрерывный"
    
    plt.suptitle(f'Детальный анализ изотермы 1900 K\nРежим: {mode_text}, Время: {t_phys:.1f} мкс', 
                fontsize=15, y=0.98)
    
    plt.tight_layout()
    plt.savefig(f'animations/isotherm_1900_analysis_{mode_text}_t{t_phys:.0f}us.png', 
               dpi=150, bbox_inches='tight')
    plt.show()