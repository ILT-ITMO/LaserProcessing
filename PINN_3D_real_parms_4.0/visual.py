import numpy as np
import matplotlib.pyplot as plt
import config
from matplotlib.animation import FuncAnimation, PillowWriter
from conditions import convert_to_physical_coords, convert_to_physical_temperature, get_physical_extent

def visualize_laser_pulses():
    """
    Визуализация временного профиля лазерных импульсов
    В зависимости от режима показывает разные графики
    """
    if config.LASER_MODE == "pulsed":
        # Для импульсного режима - гауссовы импульсы
        t_test = np.linspace(0, config.LASER_PULSE_PERIOD_NORM * 2, 1000)
        source_values = np.zeros_like(t_test)
        
        for i, t_val in enumerate(t_test):
            t_mod = t_val % config.LASER_PULSE_PERIOD_NORM
            # Гауссов импульс
            source_values[i] = config.LASER_AMPLITUDE * np.exp(-(t_mod - config.LASER_PULSE_PERIOD_NORM/2)**2 / 
                                                               (2 * config.LASER_PULSE_SIGMA_NORM**2))
        
        # Конвертация в физическое время
        t_phys = t_test * config.CHARACTERISTIC_TIME * 1e6  # в микросекунды
        
        plt.figure(figsize=(10, 4))
        plt.plot(t_phys, source_values, 'r-', linewidth=2)
        plt.xlabel('Время (мкс)')
        plt.ylabel('Относительная интенсивность')
        plt.title('Временной профиль лазерных импульсов (Гауссовы, импульсный режим)')
        plt.grid(True, alpha=0.3)
        
        # Добавим вертикальные линии для периодов
        for i in range(3):
            period_time = i * config.LASER_PULSE_PERIOD_NORM * config.CHARACTERISTIC_TIME * 1e6
            plt.axvline(x=period_time, color='gray', linestyle='--', alpha=0.5)
            plt.text(period_time + 2, 0.9, f'Импульс {i+1}', fontsize=8)
        
        plt.tight_layout()
        plt.savefig('animations/laser_pulse_profile_gaussian.png', dpi=150)
        plt.show()
        
    else:
        # Для непрерывного режима - постоянный сигнал
        t_test = np.linspace(0, config.SIMULATION_TIME_NORM, 1000)
        source_values = np.ones_like(t_test) * config.LASER_AMPLITUDE
        
        # Конвертация в физическое время
        t_phys = t_test * config.CHARACTERISTIC_TIME * 1e6  # в микросекунды
        
        plt.figure(figsize=(10, 4))
        plt.plot(t_phys, source_values, 'b-', linewidth=2)
        plt.fill_between(t_phys, 0, source_values, alpha=0.3, color='blue')
        plt.xlabel('Время (мкс)')
        plt.ylabel('Относительная интенсивность')
        plt.title('Временной профиль лазерного излучения (Непрерывный режим)')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, config.LASER_AMPLITUDE * 1.1)
        
        # Добавим информацию о мощности
        plt.text(0.02, 0.95, f'Мощность: {config.LASER_CONTINUOUS_POWER} Вт', 
                transform=plt.gca().transAxes, fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        
        plt.tight_layout()
        plt.savefig('animations/laser_continuous_profile.png', dpi=150)
        plt.show()

def visualize_laser_spatial_profile():
    """Визуализация пространственного профиля лазерного пучка"""
    x_test = np.linspace(-1, 1, 100)
    y_test = np.linspace(-1, 1, 100)
    X, Y = np.meshgrid(x_test, y_test)
    
    spatial_dist = config.LASER_AMPLITUDE * np.exp(-(X**2 + Y**2) / (config.LASER_SIGMA_NORM**2))
    
    # Конвертация в физические координаты (мкм)
    X_phys = X * config.CHARACTERISTIC_LENGTH * 1e6
    Y_phys = Y * config.CHARACTERISTIC_LENGTH * 1e6
    
    plt.figure(figsize=(8, 6))
    contour = plt.contourf(X_phys, Y_phys, spatial_dist, levels=50, cmap='hot')
    plt.colorbar(contour, label='Относительная интенсивность')
    plt.xlabel('x (мкм)')
    plt.ylabel('y (мкм)')
    
    mode_text = "СТАТИЧНЫЙ" if config.LASER_MODE == "pulsed" else "НЕПРЕРЫВНЫЙ"
    plt.title(f'Пространственное распределение лазерного пучка\n' +
              f'Радиус: {config.LASER_BEAM_RADIUS*1e6:.0f} мкм, Режим: {mode_text}')
    
    # Добавим перекрестие в центре
    plt.axhline(y=0, color='white', linestyle='--', alpha=0.5)
    plt.axvline(x=0, color='white', linestyle='--', alpha=0.5)
    plt.plot(0, 0, 'w+', markersize=10, markeredgewidth=2)
    
    plt.tight_layout()
    if config.LASER_MODE == "pulsed":
        plt.savefig('animations/laser_spatial_profile_static.png', dpi=150)
    else:
        plt.savefig('animations/laser_spatial_profile_continuous.png', dpi=150)
    plt.show()

def create_animation(U_data, x_plot, y_plot, z_plot, t_plot, title, filename):
    """
    Создает анимацию с ФИЗИЧЕСКИМИ координатами и температурой
    с добавлением срезов на разных глубинах
    Поддерживает оба режима лазера: импульсный и непрерывный
    """
    # Конвертация в физические величины
    U_physical = convert_to_physical_temperature(U_data)
    (x_phys_min, x_phys_max), (y_phys_min, y_phys_max), (z_phys_min, z_phys_max) = get_physical_extent(
        [x_plot[0], x_plot[-1]], [y_plot[0], y_plot[-1]], [z_plot[0], z_plot[-1]]
    )
    
    t_phys = t_plot * config.CHARACTERISTIC_TIME * 1e6  # мкс
    
    # Определяем индексы для срезов на глубинах 10, 20, 30 мкм
    z_phys_values = np.array(z_plot) * config.CHARACTERISTIC_LENGTH * 1e6  # все глубины в мкм
    target_depths = [10, 20, 30]  # целевые глубины в мкм
    
    depth_indices = []
    for depth in target_depths:
        idx = np.argmin(np.abs(z_phys_values - depth))
        depth_indices.append(idx)
    
    fig = plt.figure(figsize=(25, 12))
    
    def update(frame):
        fig.clear()
        
        # 1. XY срез (поверхность)
        ax1 = fig.add_subplot(3, 4, 1)
        slice_idx_xy = len(z_plot) - 1  # Поверхность (z = max)
        data_xy = U_physical[:, :, slice_idx_xy, frame].T
        
        im1 = ax1.imshow(data_xy, extent=[x_phys_min, x_phys_max, y_phys_min, y_phys_max], 
                        origin='lower', aspect='auto', cmap='hot', 
                        vmin=config.INITIAL_TEMPERATURE, vmax=np.max(U_physical))
        ax1.set_title('XY срез (поверхность)')
        ax1.set_xlabel('x (мкм)')
        ax1.set_ylabel('y (мкм)')
        # Добавим перекрестие в центре
        ax1.axhline(y=0, color='white', linestyle='--', alpha=0.5)
        ax1.axvline(x=0, color='white', linestyle='--', alpha=0.5)
        plt.colorbar(im1, ax=ax1, shrink=0.8, label='Температура (K)')
        
        # 2. XZ срез
        ax2 = fig.add_subplot(3, 4, 2)
        slice_idx_xz = len(y_plot) // 2
        data_xz = U_physical[:, slice_idx_xz, :, frame].T
        
        im2 = ax2.imshow(data_xz, extent=[x_phys_min, x_phys_max, z_phys_min, z_phys_max], 
                        origin='lower', aspect='auto', cmap='hot',
                        vmin=config.INITIAL_TEMPERATURE, vmax=np.max(U_physical))
        ax2.set_title('XZ срез (центральный)')
        ax2.set_xlabel('x (мкм)')
        ax2.set_ylabel('z (мкм)')
        # Добавляем линии на глубинах 10, 20, 30 мкм
        for depth, color in zip([10, 20, 30], ['cyan', 'lime', 'yellow']):
            ax2.axhline(y=depth, color=color, linestyle='--', alpha=0.7, linewidth=1)
        # Вертикальная линия в центре
        ax2.axvline(x=0, color='white', linestyle='--', alpha=0.5)
        plt.colorbar(im2, ax=ax2, shrink=0.8, label='Температура (K)')
        
        # 3. YZ срез
        ax3 = fig.add_subplot(3, 4, 3)
        slice_idx_yz = len(x_plot) // 2
        data_yz = U_physical[slice_idx_yz, :, :, frame].T
        
        im3 = ax3.imshow(data_yz, extent=[y_phys_min, y_phys_max, z_phys_min, z_phys_max], 
                        origin='lower', aspect='auto', cmap='hot',
                        vmin=config.INITIAL_TEMPERATURE, vmax=np.max(U_physical))
        ax3.set_title('YZ срез (центральный)')
        ax3.set_xlabel('y (мкм)')
        ax3.set_ylabel('z (мкм)')
        # Добавляем линии на глубинах 10, 20, 30 мкм
        for depth, color in zip([10, 20, 30], ['cyan', 'lime', 'yellow']):
            ax3.axhline(y=depth, color=color, linestyle='--', alpha=0.7, linewidth=1)
        # Вертикальная линия в центре
        ax3.axvline(x=0, color='white', linestyle='--', alpha=0.5)
        plt.colorbar(im3, ax=ax3, shrink=0.8, label='Температура (K)')
        
        # 4-6. XY срезы на глубинах 10, 20, 30 мкм
        colors = ['cyan', 'lime', 'yellow']
        for i, (depth_idx, depth, color) in enumerate(zip(depth_indices, target_depths, colors)):
            ax = fig.add_subplot(3, 4, 4 + i)
            data_depth = U_physical[:, :, depth_idx, frame].T
            
            actual_depth = z_phys_values[depth_idx]
            
            im = ax.imshow(data_depth, extent=[x_phys_min, x_phys_max, y_phys_min, y_phys_max], 
                          origin='lower', aspect='auto', cmap='hot',
                          vmin=config.INITIAL_TEMPERATURE, vmax=np.max(U_physical))
            ax.set_title(f'XY срез (z = {actual_depth:.1f} мкм)')
            ax.set_xlabel('x (мкм)')
            ax.set_ylabel('y (мкм)')
            # Добавляем перекрестие в центре
            ax.axhline(y=0, color='white', linestyle='--', alpha=0.5)
            ax.axvline(x=0, color='white', linestyle='--', alpha=0.5)
            # Добавляем рамку соответствующего цвета
            for spine in ax.spines.values():
                spine.set_color(color)
                spine.set_linewidth(2)
            plt.colorbar(im, ax=ax, shrink=0.8, label='Температура (K)')
        
        # 7. Временной профиль лазера (физическое время)
        ax7 = fig.add_subplot(3, 4, 7)
        
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
            
            ax7.plot(t_range_phys, laser_profile, 'r-', linewidth=1)
            ax7.axvline(x=t_phys[frame], color='blue', linestyle='--', alpha=0.7, linewidth=2)
            
            # Отметим все импульсы вертикальными линиями
            for i in range(config.NUM_PULSES + 1):
                impulse_time = i * config.LASER_PULSE_PERIOD_NORM * config.CHARACTERISTIC_TIME * 1e6
                ax7.axvline(x=impulse_time, color='gray', linestyle=':', alpha=0.5)
                if i < config.NUM_PULSES:
                    ax7.text(impulse_time + 5, 0.8, f'{i+1}', fontsize=8, ha='left')
            
            ax7.set_title(f'Лазерные импульсы ({config.NUM_PULSES} импульсов)')
            ax7.set_ylim(0, 1.1)
            
        else:
            # Непрерывный режим - постоянный сигнал
            t_range_norm = np.linspace(0, config.SIMULATION_TIME_NORM, 1000)
            t_range_phys = t_range_norm * config.CHARACTERISTIC_TIME * 1e6
            laser_profile = np.ones_like(t_range_norm) * config.LASER_AMPLITUDE
            
            ax7.plot(t_range_phys, laser_profile, 'b-', linewidth=2)
            ax7.axvline(x=t_phys[frame], color='red', linestyle='--', alpha=0.7, linewidth=2)
            ax7.fill_between(t_range_phys, 0, laser_profile, alpha=0.3, color='blue')
            
            ax7.set_title('Непрерывный лазерный источник')
            ax7.set_ylim(0, config.LASER_AMPLITUDE * 1.1)
        
        ax7.set_xlabel('Время (мкс)')
        ax7.set_ylabel('Относительная интенсивность')
        ax7.grid(True, alpha=0.3)
        ax7.set_xlim(0, max(t_phys))
        
        # 8. Информация о системе
        ax8 = fig.add_subplot(3, 4, 8)
        ax8.axis('off')
        
        current_time_phys = t_phys[frame]
        
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
        for depth_idx, depth in zip(depth_indices, target_depths):
            temp = np.max(U_physical[:, :, depth_idx, frame])
            temp_at_depths.append((depth, temp))
        
        info_text = f"Физическое время: {current_time_phys:.1f} мкс\n"
        
        if config.LASER_MODE == "pulsed":
            info_text += f"Импульс №: {pulse_number} из {config.NUM_PULSES}\n"
            info_text += f"Время в импульсе: {time_in_pulse_phys:.1f} мкс\n"
        else:
            info_text += f"Режим: НЕПРЕРЫВНЫЙ\n"
            info_text += f"Мощность: {config.LASER_CONTINUOUS_POWER} Вт\n"
        
        info_text += f"Макс. температура: {max_temp:.1f} K\n"
        info_text += f"Перегрев: {max_temp - config.INITIAL_TEMPERATURE:.1f} K\n"
        info_text += "Температуры на глубинах:\n"
        for depth, temp in temp_at_depths:
            info_text += f"  z={depth} мкм: {temp:.1f} K\n"
        info_text += f"Начальная: {config.INITIAL_TEMPERATURE} K"
        
        if config.LASER_MODE == "pulsed":
            if time_in_pulse_norm <= config.LASER_PULSE_DURATION_NORM * 2:
                info_text += "\nЛазер: АКТИВЕН"
            else:
                info_text += "\nЛазер: ВЫКЛ"
        else:
            info_text += "\nЛазер: ПОСТОЯННО ВКЛ"
        
        ax8.text(0.1, 0.5, info_text, fontsize=10, va='center', linespacing=1.4,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
        
        # 9. Пространственный профиль лазера (физические координаты)
        ax9 = fig.add_subplot(3, 4, 11)
        x_profile_norm = np.linspace(-1, 1, 100)
        x_profile_phys = x_profile_norm * config.CHARACTERISTIC_LENGTH * 1e6
        laser_spatial = config.LASER_AMPLITUDE * np.exp(-x_profile_norm**2 / (config.LASER_SIGMA_NORM**2))
        
        ax9.plot(x_profile_phys, laser_spatial, 'g-', linewidth=2)
        ax9.set_xlabel('x (мкм)')
        ax9.set_ylabel('Относительная интенсивность')
        ax9.set_title(f'Пространственный профиль пучка\n(радиус {config.LASER_BEAM_RADIUS*1e6:.0f} мкм)')
        ax9.grid(True, alpha=0.3)
        ax9.set_xlim(x_phys_min, x_phys_max)
        ax9.set_ylim(0, 1.1)
        
        # 10. График температуры по глубине
        ax10 = fig.add_subplot(3, 4, 12)
        center_x = len(x_plot) // 2
        center_y = len(y_plot) // 2
        temp_vs_depth = U_physical[center_x, center_y, :, frame]
        
        ax10.plot(temp_vs_depth, z_phys_values, 'b-', linewidth=2)
        ax10.set_xlabel('Температура (K)')
        ax10.set_ylabel('Глубина z (мкм)')
        ax10.set_title('Температура по глубине\n(в центре пучка)')
        ax10.grid(True, alpha=0.3)
        ax10.set_ylim(z_phys_min, z_phys_max)
        # Добавляем маркеры на целевых глубинах
        for depth, color in zip([10, 20, 30], ['cyan', 'lime', 'yellow']):
            depth_idx = np.argmin(np.abs(z_phys_values - depth))
            temp = temp_vs_depth[depth_idx]
            ax10.plot(temp, depth, 'o', color=color, markersize=6, markeredgecolor='black')
        
        # Супер-заголовок
        mode_text = "ИМПУЛЬСНЫЙ" if config.LASER_MODE == "pulsed" else "НЕПРЕРЫВНЫЙ"
        suptitle = f'{title}\nРежим: {mode_text}, Время: {current_time_phys:.1f} мкс'
        if config.LASER_MODE == "pulsed":
            suptitle += f' (Импульс {pulse_number}/{config.NUM_PULSES})'
        
        plt.suptitle(suptitle, fontsize=16, y=0.98)
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
    Создает график эволюции температуры в центре пучка во времени
    Поддерживает оба режима лазера
    """
    U_physical = convert_to_physical_temperature(U_data)
    t_phys = t_plot * config.CHARACTERISTIC_TIME * 1e6  # мкс
    
    # Температура в центре пучка на поверхности
    center_x = len(x_plot) // 2
    center_y = len(y_plot) // 2
    surface_z = len(z_plot) - 1
    
    center_temperature = U_physical[center_x, center_y, surface_z, :]
    
    plt.figure(figsize=(12, 6))
    
    plt.plot(t_phys, center_temperature, 'b-', linewidth=2, label='Температура в центре')
    plt.axhline(y=config.INITIAL_TEMPERATURE, color='gray', linestyle='--', 
                label=f'Начальная температура ({config.INITIAL_TEMPERATURE} K)')
    
    if config.LASER_MODE == "pulsed":
        # Отметим моменты импульсов для импульсного режима
        for i in range(config.NUM_PULSES):
            impulse_time = i * config.LASER_PULSE_PERIOD_NORM * config.CHARACTERISTIC_TIME * 1e6
            plt.axvline(x=impulse_time, color='red', linestyle=':', alpha=0.5, 
                       label='Импульс' if i == 0 else "")
            plt.text(impulse_time + 2, np.min(center_temperature) + 10, f'{i+1}', 
                    fontsize=8, color='red')
    
        title_text = f'Эволюция температуры в центре пучка\n' \
                    f'({config.NUM_PULSES} импульсов, {config.LASER_MODE} режим)'
    else:
        # Для непрерывного режима покажем только информацию о мощности
        title_text = f'Эволюция температуры в центре пучка\n' \
                    f'Непрерывный режим, Мощность: {config.LASER_CONTINUOUS_POWER} Вт'
        # Добавим заливку области нагрева
        plt.fill_between(t_phys, config.INITIAL_TEMPERATURE, center_temperature, 
                        alpha=0.3, color='blue', label='Область нагрева')
    
    plt.xlabel('Время (мкс)')
    plt.ylabel('Температура (K)')
    plt.title(title_text)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    if config.LASER_MODE == "pulsed":
        plt.savefig('animations/temperature_evolution_center_pulsed.png', dpi=150)
    else:
        plt.savefig('animations/temperature_evolution_center_continuous.png', dpi=150)
    
    plt.show()
    
    return center_temperature

def plot_depth_temperature_profiles(U_data, x_plot, y_plot, z_plot, t_plot):
    """
    Создает графики распределения температуры по глубине в разные моменты времени
    Поддерживает оба режима лазера
    """
    U_physical = convert_to_physical_temperature(U_data)
    z_phys = np.array(z_plot) * config.CHARACTERISTIC_LENGTH * 1e6  # мкм
    
    center_x = len(x_plot) // 2
    center_y = len(y_plot) // 2
    
    if config.LASER_MODE == "pulsed":
        # Для импульсного режима - моменты после каждого импульса
        key_frames = []
        for i in range(config.NUM_PULSES):
            # Момент сразу после импульса
            frame_idx = min(len(t_plot) - 1, int((i + 0.7) * len(t_plot) / config.NUM_PULSES))
            key_frames.append(frame_idx)
        
        title_suffix = f'после каждого из {config.NUM_PULSES} импульсов'
    else:
        # Для непрерывного режима - равномерные моменты времени
        key_frames = np.linspace(0, len(t_plot)-1, min(8, len(t_plot)), dtype=int)
        title_suffix = 'в разные моменты времени'
    
    plt.figure(figsize=(10, 8))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(key_frames)))
    
    for i, frame_idx in enumerate(key_frames):
        temp_profile = U_physical[center_x, center_y, :, frame_idx]
        t_phys = t_plot[frame_idx] * config.CHARACTERISTIC_TIME * 1e6
        
        if config.LASER_MODE == "pulsed":
            pulse_num = int(t_plot[frame_idx] // config.LASER_PULSE_PERIOD_NORM) + 1
            label = f'После {pulse_num} импульса ({t_phys:.0f} мкс)'
        else:
            label = f'Время {t_phys:.0f} мкс'
        
        plt.plot(temp_profile, z_phys, color=colors[i], linewidth=2, label=label)
    
    plt.xlabel('Температура (K)')
    plt.ylabel('Глубина (мкм)')
    
    mode_text = "Импульсный" if config.LASER_MODE == "pulsed" else "Непрерывный"
    plt.title(f'Распределение температуры по глубине {title_suffix}\n' +
              f'(в центре пучка, {mode_text.lower()} режим)')
    
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.gca().invert_yaxis()  # Глубина увеличивается вниз
    plt.tight_layout()
    
    if config.LASER_MODE == "pulsed":
        plt.savefig('animations/depth_temperature_profiles_pulsed.png', dpi=150)
    else:
        plt.savefig('animations/depth_temperature_profiles_continuous.png', dpi=150)
    
    plt.show()

def plot_comparison_pulse_vs_continuous(temp_pulsed, temp_continuous, t_plot, x_label="Время (мкс)"):
    """
    Создает график сравнения температуры для импульсного и непрерывного режимов
    
    Args:
        temp_pulsed: температура для импульсного режима (массив)
        temp_continuous: температура для непрерывного режима (массив)
        t_plot: временная ось (безразмерная)
        x_label: подпись оси X
    """
    if temp_pulsed is None or temp_continuous is None:
        print("Для сравнения нужны данные обоих режимов!")
        return
    
    t_phys = t_plot * config.CHARACTERISTIC_TIME * 1e6  # мкс
    
    plt.figure(figsize=(12, 7))
    
    # Графики температуры
    plt.plot(t_phys, temp_pulsed, 'r-', linewidth=2, label='Импульсный режим', alpha=0.8)
    plt.plot(t_phys, temp_continuous, 'b-', linewidth=2, label='Непрерывный режим', alpha=0.8)
    
    # Начальная температура
    plt.axhline(y=config.INITIAL_TEMPERATURE, color='gray', linestyle='--', 
                alpha=0.5, label=f'Начальная ({config.INITIAL_TEMPERATURE} K)')
    
    # Заполнение между кривыми для наглядности
    plt.fill_between(t_phys, temp_pulsed, temp_continuous, 
                     where=temp_pulsed >= temp_continuous, 
                     color='red', alpha=0.2, label='Импульсный > Непрерывный')
    plt.fill_between(t_phys, temp_pulsed, temp_continuous, 
                     where=temp_pulsed < temp_continuous, 
                     color='blue', alpha=0.2, label='Импульсный < Непрерывный')
    
    # Максимальные значения
    max_pulsed = np.max(temp_pulsed)
    max_continuous = np.max(temp_continuous)
    max_time_pulsed = t_phys[np.argmax(temp_pulsed)]
    max_time_continuous = t_phys[np.argmax(temp_continuous)]
    
    plt.scatter([max_time_pulsed], [max_pulsed], color='red', s=100, zorder=5,
                label=f'Макс. импульсный: {max_pulsed:.1f} K')
    plt.scatter([max_time_continuous], [max_continuous], color='blue', s=100, zorder=5,
                label=f'Макс. непрерывный: {max_continuous:.1f} K')
    
    # Информация в текстовом блоке
    info_text = f"Сравнение режимов лазерного нагрева:\n"
    info_text += f"• Импульсный: {max_pulsed:.1f} K ({max_time_pulsed:.0f} мкс)\n"
    info_text += f"• Непрерывный: {max_continuous:.1f} K ({max_time_continuous:.0f} мкс)\n"
    info_text += f"• Разница: {abs(max_pulsed - max_continuous):.1f} K\n"
    if max_pulsed > max_continuous:
        info_text += f"• Импульсный горячее на {(max_pulsed/max_continuous-1)*100:.1f}%"
    else:
        info_text += f"• Непрерывный горячее на {(max_continuous/max_pulsed-1)*100:.1f}%"
    
    plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
    
    plt.xlabel(x_label)
    plt.ylabel('Температура (K)')
    plt.title('Сравнение температурных профилей: Импульсный vs Непрерывный режим')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig('animations/comparison_pulse_vs_continuous.png', dpi=150)
    plt.show()

def plot_temperature_distribution_at_time(U_data, x_plot, y_plot, z_plot, t_plot, time_idx=-1):
    """
    Создает 3D-подобную визуализацию распределения температуры в выбранный момент времени
    
    Args:
        time_idx: индекс времени (по умолчанию последний момент)
    """
    U_physical = convert_to_physical_temperature(U_data)
    x_phys = np.array(x_plot) * config.CHARACTERISTIC_LENGTH * 1e6
    y_phys = np.array(y_plot) * config.CHARACTERISTIC_LENGTH * 1e6
    z_phys = np.array(z_plot) * config.CHARACTERISTIC_LENGTH * 1e6
    
    t_phys = t_plot[time_idx] * config.CHARACTERISTIC_TIME * 1e6
    
    fig = plt.figure(figsize=(18, 10))
    
    # 1. 3D поверхность температуры на поверхности
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    X, Y = np.meshgrid(x_phys, y_phys)
    Z_surface = U_physical[:, :, -1, time_idx].T  # Поверхность (z = max)
    
    surf = ax1.plot_surface(X, Y, Z_surface, cmap='hot', alpha=0.8)
    ax1.set_xlabel('x (мкм)')
    ax1.set_ylabel('y (мкм)')
    ax1.set_zlabel('Температура (K)')
    ax1.set_title(f'Температура на поверхности\nВремя: {t_phys:.1f} мкс')
    fig.colorbar(surf, ax=ax1, shrink=0.6, label='Температура (K)')
    
    # 2. Изотермы на поверхности
    ax2 = fig.add_subplot(2, 3, 2)
    contour = ax2.contourf(X, Y, Z_surface, levels=20, cmap='hot')
    ax2.contour(X, Y, Z_surface, levels=10, colors='white', linewidths=0.5, alpha=0.5)
    ax2.set_xlabel('x (мкм)')
    ax2.set_ylabel('y (мкм)')
    ax2.set_title('Изотермы на поверхности')
    ax2.grid(True, alpha=0.3)
    fig.colorbar(contour, ax=ax2, shrink=0.9, label='Температура (K)')
    
    # 3. Распределение по глубине в центре
    ax3 = fig.add_subplot(2, 3, 3)
    center_x = len(x_plot) // 2
    center_y = len(y_plot) // 2
    temp_profile = U_physical[center_x, center_y, :, time_idx]
    
    ax3.plot(temp_profile, z_phys, 'b-', linewidth=2)
    ax3.set_xlabel('Температура (K)')
    ax3.set_ylabel('Глубина z (мкм)')
    ax3.set_title('Распределение по глубине (центр)')
    ax3.grid(True, alpha=0.3)
    ax3.invert_yaxis()  # Глубина увеличивается вниз
    
    # Отметим температуры на глубинах 10, 20, 30 мкм
    target_depths = [10, 20, 30]
    colors = ['cyan', 'lime', 'yellow']
    
    for depth, color in zip(target_depths, colors):
        depth_idx = np.argmin(np.abs(z_phys - depth))
        temp = temp_profile[depth_idx]
        ax3.plot(temp, depth, 'o', color=color, markersize=8, markeredgecolor='black')
        ax3.text(temp + 5, depth, f'{temp:.0f}K', fontsize=8, va='center')
    
    # 4. Горизонтальный срез на глубине 20 мкм
    ax4 = fig.add_subplot(2, 3, 4)
    depth_idx = np.argmin(np.abs(z_phys - 20))
    depth_slice = U_physical[:, :, depth_idx, time_idx].T
    
    im4 = ax4.imshow(depth_slice, extent=[x_phys[0], x_phys[-1], y_phys[0], y_phys[-1]], 
                    origin='lower', aspect='auto', cmap='hot')
    ax4.set_xlabel('x (мкм)')
    ax4.set_ylabel('y (мкм)')
    ax4.set_title(f'Срез на глубине {z_phys[depth_idx]:.0f} мкм')
    ax4.grid(True, alpha=0.3)
    fig.colorbar(im4, ax=ax4, shrink=0.9, label='Температура (K)')
    
    # 5. Профиль вдоль оси X на поверхности
    ax5 = fig.add_subplot(2, 3, 5)
    y_center_idx = len(y_plot) // 2
    x_profile = U_physical[:, y_center_idx, -1, time_idx]
    
    ax5.plot(x_phys, x_profile, 'r-', linewidth=2, label='Поверхность (y=0)')
    ax5.fill_between(x_phys, config.INITIAL_TEMPERATURE, x_profile, alpha=0.3, color='red')
    ax5.set_xlabel('x (мкм)')
    ax5.set_ylabel('Температура (K)')
    ax5.set_title('Профиль температуры вдоль оси X')
    ax5.grid(True, alpha=0.3)
    ax5.legend()
    
    # 6. Информационная панель
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    
    max_temp = np.max(U_physical[:, :, :, time_idx])
    min_temp = np.min(U_physical[:, :, :, time_idx])
    avg_temp = np.mean(U_physical[:, :, :, time_idx])
    
    info_text = f"АНАЛИЗ ТЕМПЕРАТУРНОГО ПОЛЯ\n"
    info_text += "=" * 30 + "\n"
    info_text += f"Время: {t_phys:.1f} мкс\n\n"
    info_text += f"Максимальная температура: {max_temp:.1f} K\n"
    info_text += f"Минимальная температура: {min_temp:.1f} K\n"
    info_text += f"Средняя температура: {avg_temp:.1f} K\n"
    info_text += f"Перегрев: {max_temp - config.INITIAL_TEMPERATURE:.1f} K\n\n"
    
    if config.LASER_MODE == "pulsed":
        pulse_number = min(int(t_plot[time_idx] // config.LASER_PULSE_PERIOD_NORM) + 1, config.NUM_PULSES)
        info_text += f"РЕЖИМ: ИМПУЛЬСНЫЙ\n"
        info_text += f"Импульс: {pulse_number}/{config.NUM_PULSES}\n"
    else:
        info_text += f"РЕЖИМ: НЕПРЕРЫВНЫЙ\n"
        info_text += f"Мощность: {config.LASER_CONTINUOUS_POWER} Вт\n"
    
    info_text += "\nТемпература на глубинах:\n"
    for depth, color in zip(target_depths, colors):
        depth_idx = np.argmin(np.abs(z_phys - depth))
        temp = U_physical[center_x, center_y, depth_idx, time_idx]
        info_text += f"  {depth} мкм: {temp:.1f} K\n"
    
    ax6.text(0.1, 0.95, info_text, fontsize=10, va='top',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    mode_text = "Импульсный" if config.LASER_MODE == "pulsed" else "Непрерывный"
    plt.suptitle(f'Детальный анализ температурного поля\nРежим: {mode_text}, Время: {t_phys:.1f} мкс', 
                fontsize=14, y=0.98)
    
    plt.tight_layout()
    
    if config.LASER_MODE == "pulsed":
        plt.savefig(f'animations/temperature_field_analysis_pulsed_t{t_phys:.0f}us.png', dpi=150, bbox_inches='tight')
    else:
        plt.savefig(f'animations/temperature_field_analysis_continuous_t{t_phys:.0f}us.png', dpi=150, bbox_inches='tight')
    
    plt.show()

def plot_laser_intensity_3d():
    """Создает 3D визуализацию интенсивности лазерного пучка"""
    x = np.linspace(-1, 1, 100) * config.CHARACTERISTIC_LENGTH * 1e6
    y = np.linspace(-1, 1, 100) * config.CHARACTERISTIC_LENGTH * 1e6
    X, Y = np.meshgrid(x, y)
    
    # Вычисляем интенсивность в относительных единицах
    intensity = np.exp(-(X**2 + Y**2) / (config.LASER_BEAM_RADIUS*1e6)**2)
    
    fig = plt.figure(figsize=(14, 6))
    
    # 3D поверхность
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    surf = ax1.plot_surface(X, Y, intensity, cmap='hot', alpha=0.8, 
                           rstride=2, cstride=2, linewidth=0.5, antialiased=True)
    ax1.set_xlabel('x (мкм)')
    ax1.set_ylabel('y (мкм)')
    ax1.set_zlabel('Относительная интенсивность')
    ax1.set_title('3D профиль лазерного пучка')
    fig.colorbar(surf, ax=ax1, shrink=0.6, label='Интенсивность')
    
    # 2D контурный график
    ax2 = fig.add_subplot(1, 2, 2)
    contour = ax2.contourf(X, Y, intensity, levels=20, cmap='hot')
    ax2.contour(X, Y, intensity, levels=10, colors='white', linewidths=0.5, alpha=0.5)
    
    # Добавим круги на 1/e² радиусах
    radius = config.LASER_BEAM_RADIUS * 1e6
    circle = plt.Circle((0, 0), radius, fill=False, color='cyan', linestyle='--', 
                       linewidth=2, alpha=0.7, label=f'Радиус пучка: {radius:.0f} мкм')
    ax2.add_patch(circle)
    
    circle_sigma = plt.Circle((0, 0), radius/1.414, fill=False, color='yellow', 
                            linestyle=':', linewidth=2, alpha=0.7, label=f'1σ: {radius/1.414:.0f} мкм')
    ax2.add_patch(circle_sigma)
    
    ax2.set_xlabel('x (мкм)')
    ax2.set_ylabel('y (мкм)')
    ax2.set_title('2D распределение интенсивности')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_aspect('equal')
    fig.colorbar(contour, ax=ax2, shrink=0.9, label='Интенсивность')
    
    mode_text = "Импульсный" if config.LASER_MODE == "pulsed" else "Непрерывный"
    power_text = f"{config.LASER_AVG_POWER} Вт (ср.)" if config.LASER_MODE == "pulsed" else f"{config.LASER_CONTINUOUS_POWER} Вт"
    
    plt.suptitle(f'Лазерный пучок: {mode_text} режим, {power_text}, λ={config.LASER_WAVELENGTH*1e6:.1f} мкм',
                fontsize=14, y=0.95)
    
    plt.tight_layout()
    plt.savefig('animations/laser_intensity_3d_profile.png', dpi=150, bbox_inches='tight')
    plt.show()