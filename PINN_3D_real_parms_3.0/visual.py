# import numpy as np
# import matplotlib.pyplot as plt
# import config
# from matplotlib.animation import FuncAnimation, PillowWriter
# from conditions import convert_to_physical_coords, convert_to_physical_temperature, get_physical_extent

# def visualize_laser_pulses():
#     """Визуализация временного профиля лазерных импульсов (ГАУССОВЫХ)"""
#     t_test = np.linspace(0, config.LASER_PULSE_PERIOD_NORM * 2, 1000)
#     source_values = np.zeros_like(t_test)
    
#     for i, t_val in enumerate(t_test):
#         t_mod = t_val % config.LASER_PULSE_PERIOD_NORM
#         # Гауссов импульс
#         source_values[i] = config.LASER_AMPLITUDE * np.exp(-(t_mod - config.LASER_PULSE_SIGMA_NORM * 3)**2 / (2 * config.LASER_PULSE_SIGMA_NORM**2))
    
#     # Конвертация в физическое время
#     t_phys = t_test * config.CHARACTERISTIC_TIME * 1e6  # в микросекунды
    
#     plt.figure(figsize=(10, 4))
#     plt.plot(t_phys, source_values)
#     plt.xlabel('Время (мкс)')
#     plt.ylabel('Относительная интенсивность')
#     plt.title('Временной профиль лазерных импульсов (Гауссов)')
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig('animations/laser_pulse_profile_gaussian.png')
#     plt.show()

# def visualize_laser_spatial_profile():
#     """Визуализация пространственного профиля лазерного пучка"""
#     x_test = np.linspace(-1, 1, 100)
#     y_test = np.linspace(-1, 1, 100)
#     X, Y = np.meshgrid(x_test, y_test)
    
#     spatial_dist = config.LASER_AMPLITUDE * np.exp(-(X**2 + Y**2) / (2 * config.LASER_SIGMA_NORM**2))
    
#     # Конвертация в физические координаты (мкм)
#     X_phys = X * config.CHARACTERISTIC_LENGTH * 1e6
#     Y_phys = Y * config.CHARACTERISTIC_LENGTH * 1e6
    
#     plt.figure(figsize=(8, 6))
#     contour = plt.contourf(X_phys, Y_phys, spatial_dist, levels=50, cmap='hot')
#     plt.colorbar(contour, label='Относительная интенсивность')
#     plt.xlabel('x (мкм)')
#     plt.ylabel('y (мкм)')
#     plt.title(f'Пространственное распределение лазерного пучка\n(радиус {config.LASER_BEAM_RADIUS*1e6:.0f} мкм)')
#     plt.tight_layout()
#     plt.savefig('animations/laser_spatial_profile_physical.png')
#     plt.show()

# def create_animation(U_data, x_plot, y_plot, z_plot, t_plot, title, filename):
#     """
#     Создает анимацию с ФИЗИЧЕСКИМИ координатами и температурой
#     с добавлением срезов на разных глубинах
#     """
#     # Конвертация в физические величины
#     U_physical = convert_to_physical_temperature(U_data)
#     (x_phys_min, x_phys_max), (y_phys_min, y_phys_max), (z_phys_min, z_phys_max) = get_physical_extent(
#         [x_plot[0], x_plot[-1]], [y_plot[0], y_plot[-1]], [z_plot[0], z_plot[-1]]
#     )
    
#     t_phys = t_plot * config.CHARACTERISTIC_TIME * 1e6  # мкс
    
#     # Определяем индексы для срезов на глубинах 10, 20, 30 мкм
#     z_phys_values = np.array(z_plot) * config.CHARACTERISTIC_LENGTH * 1e6  # все глубины в мкм
#     target_depths = [10, 20, 30]  # целевые глубины в мкм
    
#     depth_indices = []
#     for depth in target_depths:
#         idx = np.argmin(np.abs(z_phys_values - depth))
#         depth_indices.append(idx)
    
#     fig = plt.figure(figsize=(25, 12))
    
#     def update(frame):
#         fig.clear()
        
#         # 1. XY срез (поверхность)
#         ax1 = fig.add_subplot(3, 4, 1)
#         slice_idx_xy = len(z_plot) - 1  # Поверхность (z = max)
#         data_xy = U_physical[:, :, slice_idx_xy, frame].T
        
#         im1 = ax1.imshow(data_xy, extent=[x_phys_min, x_phys_max, y_phys_min, y_phys_max], 
#                         origin='lower', aspect='auto', cmap='hot', 
#                         vmin=config.INITIAL_TEMPERATURE, vmax=np.max(U_physical))
#         ax1.set_title('XY срез (поверхность)')
#         ax1.set_xlabel('x (мкм)')
#         ax1.set_ylabel('y (мкм)')
#         plt.colorbar(im1, ax=ax1, shrink=0.8, label='Температура (K)')
        
#         # 2. XZ срез
#         ax2 = fig.add_subplot(3, 4, 2)
#         slice_idx_xz = len(y_plot) // 2
#         data_xz = U_physical[:, slice_idx_xz, :, frame].T
        
#         im2 = ax2.imshow(data_xz, extent=[x_phys_min, x_phys_max, z_phys_min, z_phys_max], 
#                         origin='lower', aspect='auto', cmap='hot',
#                         vmin=config.INITIAL_TEMPERATURE, vmax=np.max(U_physical))
#         ax2.set_title('XZ срез (центральный)')
#         ax2.set_xlabel('x (мкм)')
#         ax2.set_ylabel('z (мкм)')
#         # Добавляем линии на глубинах 10, 20, 30 мкм
#         for depth, color in zip([10, 20, 30], ['cyan', 'lime', 'yellow']):
#             ax2.axhline(y=depth, color=color, linestyle='--', alpha=0.7, linewidth=1)
#         plt.colorbar(im2, ax=ax2, shrink=0.8, label='Температура (K)')
        
#         # 3. YZ срез
#         ax3 = fig.add_subplot(3, 4, 3)
#         slice_idx_yz = len(x_plot) // 2
#         data_yz = U_physical[slice_idx_yz, :, :, frame].T
        
#         im3 = ax3.imshow(data_yz, extent=[y_phys_min, y_phys_max, z_phys_min, z_phys_max], 
#                         origin='lower', aspect='auto', cmap='hot',
#                         vmin=config.INITIAL_TEMPERATURE, vmax=np.max(U_physical))
#         ax3.set_title('YZ срез (центральный)')
#         ax3.set_xlabel('y (мкм)')
#         ax3.set_ylabel('z (мкм)')
#         # Добавляем линии на глубинах 10, 20, 30 мкм
#         for depth, color in zip([10, 20, 30], ['cyan', 'lime', 'yellow']):
#             ax3.axhline(y=depth, color=color, linestyle='--', alpha=0.7, linewidth=1)
#         plt.colorbar(im3, ax=ax3, shrink=0.8, label='Температура (K)')
        
#         # 4-6. XY срезы на глубинах 10, 20, 30 мкм
#         colors = ['cyan', 'lime', 'yellow']
#         for i, (depth_idx, depth, color) in enumerate(zip(depth_indices, target_depths, colors)):
#             ax = fig.add_subplot(3, 4, 4 + i)
#             data_depth = U_physical[:, :, depth_idx, frame].T
            
#             actual_depth = z_phys_values[depth_idx]
            
#             im = ax.imshow(data_depth, extent=[x_phys_min, x_phys_max, y_phys_min, y_phys_max], 
#                           origin='lower', aspect='auto', cmap='hot',
#                           vmin=config.INITIAL_TEMPERATURE, vmax=np.max(U_physical))
#             ax.set_title(f'XY срез (z = {actual_depth:.1f} мкм)')
#             ax.set_xlabel('x (мкм)')
#             ax.set_ylabel('y (мкм)')
#             # Добавляем рамку соответствующего цвета
#             for spine in ax.spines.values():
#                 spine.set_color(color)
#                 spine.set_linewidth(2)
#             plt.colorbar(im, ax=ax, shrink=0.8, label='Температура (K)')
        
#         # 7. Временной профиль лазера (физическое время)
#         ax7 = fig.add_subplot(3, 4, 7)
#         t_range_norm = np.linspace(0, max(t_plot), 1000)
#         t_range_phys = t_range_norm * config.CHARACTERISTIC_TIME * 1e6  # мкс
#         laser_profile = np.zeros_like(t_range_norm)
        
#         for i, t_val in enumerate(t_range_norm):
#             t_mod = t_val % config.LASER_PULSE_PERIOD_NORM
#             # Гауссов импульс
#             laser_profile[i] = config.LASER_AMPLITUDE * np.exp(-(t_mod - config.LASER_PULSE_SIGMA_NORM * 3)**2 / (2 * config.LASER_PULSE_SIGMA_NORM**2))
        
#         # ax7.plot(t_range_phys, laser_profile, 'r-', linewidth=2)
#         # ax7.axvline(x=t_phys[frame], color='blue', linestyle='--', alpha=0.7, linewidth=2)
#         # ax7.set_xlabel('Время (мкс)')
#         # ax7.set_ylabel('Относительная интенсивность')
#         # ax7.set_title('Лазерные импульсы (Гауссовы)')
#         # ax7.grid(True)
#         # ax7.set_xlim(0, max(t_range_phys))
#         ax7.plot(t_range_phys, laser_profile, 'r-', linewidth=2)
#         ax7.axvline(x=t_phys[frame], color='blue', linestyle='--', alpha=0.7, linewidth=2)
#         ax7.set_xlabel('Время (мкс)')
#         ax7.set_ylabel('Относительная интенсивность')
#         ax7.set_title('Лазерные импульсы (Гауссовы)\nСТАТИЧНЫЙ пучок')  # измените заголовок
#         ax7.grid(True)
#         ax7.set_xlim(0, max(t_range_phys))

        
#         # 8. Информация о системе
#         ax8 = fig.add_subplot(3, 4, 8)
#         ax8.axis('off')
        
#         current_time_phys = t_phys[frame]
#         pulse_number = int(t_plot[frame] // config.LASER_PULSE_PERIOD_NORM) + 1
#         time_in_pulse_norm = t_plot[frame] % config.LASER_PULSE_PERIOD_NORM
#         time_in_pulse_phys = time_in_pulse_norm * config.CHARACTERISTIC_TIME * 1e6
        
#         max_temp = np.max(U_physical[:, :, :, frame])
#         min_temp = np.min(U_physical[:, :, :, frame])
        
#         # Температуры на разных глубинах
#         temp_at_depths = []
#         for depth_idx, depth in zip(depth_indices, target_depths):
#             temp = np.max(U_physical[:, :, depth_idx, frame])
#             temp_at_depths.append((depth, temp))
        
#         # info_text = f"Физическое время: {current_time_phys:.1f} мкс\n"
#         # info_text += f"Импульс №: {pulse_number}\n"
#         # info_text += f"Время в импульсе: {time_in_pulse_phys:.1f} мкс\n"
#         # info_text += f"Макс. температура: {max_temp:.1f} K\n"
#         # info_text += f"Перегрев: {max_temp - config.INITIAL_TEMPERATURE:.1f} K\n"
#         # info_text += "Температуры на глубинах:\n"
#         # for depth, temp in temp_at_depths:
#         #     info_text += f"  z={depth} мкм: {temp:.1f} K\n"
#         # info_text += f"Начальная: {config.INITIAL_TEMPERATURE} K"
        
#         # if time_in_pulse_norm <= config.LASER_PULSE_DURATION_NORM * 2:
#         #     info_text += "\nЛазер: АКТИВЕН"
#         # else:
#         #     info_text += "\nЛазер: ВЫКЛ"
        
#         # ax8.text(0.1, 0.5, info_text, fontsize=10, va='center', linespacing=1.4)
#         info_text = f"Физическое время: {current_time_phys:.1f} мкс\n"
#         info_text += f"Импульс №: {pulse_number}\n"
#         info_text += f"Время в импульсе: {time_in_pulse_phys:.1f} мкс\n"
#         info_text += f"Макс. температура: {max_temp:.1f} K\n"
#         info_text += f"Перегрев: {max_temp - config.INITIAL_TEMPERATURE:.1f} K\n"
#         info_text += "Температуры на глубинах:\n"
#         for depth, temp in temp_at_depths:
#             info_text += f"  z={depth} мкм: {temp:.1f} K\n"
#         info_text += f"Начальная: {config.INITIAL_TEMPERATURE} K"
#         info_text += "\nРежим: СТАТИЧНЫЙ пучок"  # добавим информацию о режиме
#         ax8.text(0.1, 0.5, info_text, fontsize=10, va='center', linespacing=1.4)


#         # 9. Пространственный профиль лазера (физические координаты)
#         ax9 = fig.add_subplot(3, 4, 11)
#         x_profile_norm = np.linspace(-1, 1, 100)
#         x_profile_phys = x_profile_norm * config.CHARACTERISTIC_LENGTH * 1e6
#         laser_spatial = config.LASER_AMPLITUDE * np.exp(-x_profile_norm**2 / (2 * config.LASER_SIGMA_NORM**2))
        
#         ax9.plot(x_profile_phys, laser_spatial, 'g-', linewidth=2)
#         ax9.set_xlabel('x (мкм)')
#         ax9.set_ylabel('Относительная интенсивность')
#         ax9.set_title(f'Профиль пучка ({config.LASER_BEAM_RADIUS*1e6:.0f} мкм)')
#         ax9.grid(True)
#         ax9.set_xlim(x_phys_min, x_phys_max)
        
#         # 10. График температуры по глубине
#         ax10 = fig.add_subplot(3, 4, 12)
#         center_x = len(x_plot) // 2
#         center_y = len(y_plot) // 2
#         temp_vs_depth = U_physical[center_x, center_y, :, frame]
        
#         ax10.plot(temp_vs_depth, z_phys_values, 'b-', linewidth=2)
#         ax10.set_xlabel('Температура (K)')
#         ax10.set_ylabel('Глубина z (мкм)')
#         ax10.set_title('Температура по глубине\n(в центре пучка)')
#         ax10.grid(True)
#         ax10.set_ylim(z_phys_min, z_phys_max)
#         # Добавляем маркеры на целевых глубинах
#         for depth, color in zip([10, 20, 30], ['cyan', 'lime', 'yellow']):
#             depth_idx = np.argmin(np.abs(z_phys_values - depth))
#             temp = temp_vs_depth[depth_idx]
#             ax10.plot(temp, depth, 'o', color=color, markersize=6, markeredgecolor='black')
        
#         plt.suptitle(f'{title}\nФизическое время: {current_time_phys:.1f} мкс', fontsize=16)
#         plt.tight_layout()
#         return fig,

#     ani = FuncAnimation(fig, update, frames=len(t_plot), interval=500, blit=False, repeat=True)
    
#     writer = PillowWriter(fps=2)
#     ani.save(filename, writer=writer, dpi=100)
#     plt.close(fig)
#     return ani



# !!! ВЕРСИЯ 2.0 - визуализация тут вся вайбкодинг так что.... не знаю стоит ли копаться в этом коде, думаю, там грязно !!!

import numpy as np
import matplotlib.pyplot as plt
import config
from matplotlib.animation import FuncAnimation, PillowWriter
from conditions import convert_to_physical_coords, convert_to_physical_temperature, get_physical_extent

def visualize_laser_pulses():
    """Визуализация временного профиля лазерных импульсов (ГАУССОВЫХ)"""
    t_test = np.linspace(0, config.LASER_PULSE_PERIOD_NORM * 2, 1000)
    source_values = np.zeros_like(t_test)
    
    for i, t_val in enumerate(t_test):
        t_mod = t_val % config.LASER_PULSE_PERIOD_NORM
        # Гауссов импульс
        source_values[i] = config.LASER_AMPLITUDE * np.exp(-(t_mod - config.LASER_PULSE_PERIOD_NORM/2)**2 / (2 * config.LASER_PULSE_SIGMA_NORM**2))
    
    # Конвертация в физическое время
    t_phys = t_test * config.CHARACTERISTIC_TIME * 1e6  # в микросекунды
    
    plt.figure(figsize=(10, 4))
    plt.plot(t_phys, source_values, 'r-', linewidth=2)
    plt.xlabel('Время (мкс)')
    plt.ylabel('Относительная интенсивность')
    plt.title('Временной профиль лазерных импульсов (Гауссов)')
    plt.grid(True, alpha=0.3)
    
    # Добавим вертикальные линии для периодов
    for i in range(3):
        period_time = i * config.LASER_PULSE_PERIOD_NORM * config.CHARACTERISTIC_TIME * 1e6
        plt.axvline(x=period_time, color='gray', linestyle='--', alpha=0.5)
        plt.text(period_time + 2, 0.9, f'Импульс {i+1}', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('animations/laser_pulse_profile_gaussian.png', dpi=150)
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
    plt.title(f'Пространственное распределение лазерного пучка\n(радиус {config.LASER_BEAM_RADIUS*1e6:.0f} мкм, СТАТИЧНЫЙ)')
    
    # Добавим перекрестие в центре
    plt.axhline(y=0, color='white', linestyle='--', alpha=0.5)
    plt.axvline(x=0, color='white', linestyle='--', alpha=0.5)
    plt.plot(0, 0, 'w+', markersize=10, markeredgewidth=2)
    
    plt.tight_layout()
    plt.savefig('animations/laser_spatial_profile_static.png', dpi=150)
    plt.show()

def create_animation(U_data, x_plot, y_plot, z_plot, t_plot, title, filename):
    """
    Создает анимацию с ФИЗИЧЕСКИМИ координатами и температурой
    с добавлением срезов на разных глубинах для 8 импульсов
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
        
        # 7. Временной профиль лазера (физическое время) - ОБНОВЛЕННЫЙ ДЛЯ 8 ИМПУЛЬСОВ
        ax7 = fig.add_subplot(3, 4, 7)
        
        # Временной диапазон для всех 8 импульсов
        t_range_norm = np.linspace(0, config.SIMULATION_TIME_NORM, 1000)
        t_range_phys = t_range_norm * config.CHARACTERISTIC_TIME * 1e6  # мкс
        laser_profile = np.zeros_like(t_range_norm)
        
        for i, t_val in enumerate(t_range_norm):
            t_mod = t_val % config.LASER_PULSE_PERIOD_NORM
            # Гауссов импульс
            laser_profile[i] = config.LASER_AMPLITUDE * np.exp(-(t_mod - config.LASER_PULSE_PERIOD_NORM/2)**2 / (2 * config.LASER_PULSE_SIGMA_NORM**2))
        
        ax7.plot(t_range_phys, laser_profile, 'r-', linewidth=1)
        ax7.axvline(x=t_phys[frame], color='blue', linestyle='--', alpha=0.7, linewidth=2)
        
        # Отметим все 8 импульсов вертикальными линиями
        for i in range(config.NUM_PULSES + 1):
            impulse_time = i * config.LASER_PULSE_PERIOD_NORM * config.CHARACTERISTIC_TIME * 1e6
            ax7.axvline(x=impulse_time, color='gray', linestyle=':', alpha=0.5)
            if i < config.NUM_PULSES:
                ax7.text(impulse_time + 5, 0.8, f'{i+1}', fontsize=8, ha='left')
        
        ax7.set_xlabel('Время (мкс)')
        ax7.set_ylabel('Относительная интенсивность')
        ax7.set_title(f'Лазерные импульсы ({config.NUM_PULSES} импульсов)')
        ax7.grid(True, alpha=0.3)
        ax7.set_xlim(0, max(t_range_phys))
        ax7.set_ylim(0, 1.1)
        
        # 8. Информация о системе - ОБНОВЛЕННАЯ ДЛЯ 8 ИМПУЛЬСОВ
        ax8 = fig.add_subplot(3, 4, 8)
        ax8.axis('off')
        
        current_time_phys = t_phys[frame]
        pulse_number = int(t_plot[frame] // config.LASER_PULSE_PERIOD_NORM) + 1
        time_in_pulse_norm = t_plot[frame] % config.LASER_PULSE_PERIOD_NORM
        time_in_pulse_phys = time_in_pulse_norm * config.CHARACTERISTIC_TIME * 1e6
        
        max_temp = np.max(U_physical[:, :, :, frame])
        min_temp = np.min(U_physical[:, :, :, frame])
        
        # Температуры на разных глубинах
        temp_at_depths = []
        for depth_idx, depth in zip(depth_indices, target_depths):
            temp = np.max(U_physical[:, :, depth_idx, frame])
            temp_at_depths.append((depth, temp))
        
        info_text = f"Физическое время: {current_time_phys:.1f} мкс\n"
        info_text += f"Импульс №: {pulse_number} из {config.NUM_PULSES}\n"
        info_text += f"Время в импульсе: {time_in_pulse_phys:.1f} мкс\n"
        info_text += f"Макс. температура: {max_temp:.1f} K\n"
        info_text += f"Перегрев: {max_temp - config.INITIAL_TEMPERATURE:.1f} K\n"
        info_text += "Температуры на глубинах:\n"
        for depth, temp in temp_at_depths:
            info_text += f"  z={depth} мкм: {temp:.1f} K\n"
        info_text += f"Начальная: {config.INITIAL_TEMPERATURE} K"
        
        if time_in_pulse_norm <= config.LASER_PULSE_DURATION_NORM * 2:
            info_text += "\nЛазер: АКТИВЕН"
        else:
            info_text += "\nЛазер: ВЫКЛ"
        
        info_text += f"\nРежим: СТАТИЧНЫЙ пучок"
        
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
        ax9.set_title(f'Профиль пучка ({config.LASER_BEAM_RADIUS*1e6:.0f} мкм)')
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
        
        plt.suptitle(f'{title}\nФизическое время: {current_time_phys:.1f} мкс (Импульс {pulse_number}/{config.NUM_PULSES})', 
                    fontsize=16, y=0.98)
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
    
    # Отметим моменты импульсов
    for i in range(config.NUM_PULSES):
        impulse_time = i * config.LASER_PULSE_PERIOD_NORM * config.CHARACTERISTIC_TIME * 1e6
        plt.axvline(x=impulse_time, color='red', linestyle=':', alpha=0.5, label='Импульс' if i == 0 else "")
        plt.text(impulse_time + 2, np.min(center_temperature) + 10, f'{i+1}', 
                fontsize=8, color='red')
    
    plt.xlabel('Время (мкс)')
    plt.ylabel('Температура (K)')
    plt.title(f'Эволюция температуры в центре пучка\n({config.NUM_PULSES} импульсов, статичный пучок)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('animations/temperature_evolution_center.png', dpi=150)
    plt.show()
    
    return center_temperature

def plot_depth_temperature_profiles(U_data, x_plot, y_plot, z_plot, t_plot):
    """
    Создает графики распределения температуры по глубине в разные моменты времени
    """
    U_physical = convert_to_physical_temperature(U_data)
    z_phys = np.array(z_plot) * config.CHARACTERISTIC_LENGTH * 1e6  # мкм
    
    center_x = len(x_plot) // 2
    center_y = len(y_plot) // 2
    
    # Выберем ключевые моменты времени (после каждого импульса)
    key_frames = []
    for i in range(config.NUM_PULSES):
        # Момент сразу после импульса
        frame_idx = min(len(t_plot) - 1, int((i + 0.7) * len(t_plot) / config.NUM_PULSES))
        key_frames.append(frame_idx)
    
    plt.figure(figsize=(10, 8))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(key_frames)))
    
    for i, frame_idx in enumerate(key_frames):
        temp_profile = U_physical[center_x, center_y, :, frame_idx]
        t_phys = t_plot[frame_idx] * config.CHARACTERISTIC_TIME * 1e6
        pulse_num = int(t_plot[frame_idx] // config.LASER_PULSE_PERIOD_NORM) + 1
        
        plt.plot(temp_profile, z_phys, color=colors[i], linewidth=2, 
                label=f'После {pulse_num} импульса ({t_phys:.0f} мкс)')
    
    plt.xlabel('Температура (K)')
    plt.ylabel('Глубина (мкм)')
    plt.title('Распределение температуры по глубине\nпосле каждого импульса (в центре пучка)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.gca().invert_yaxis()  # Глубина увеличивается вниз
    plt.tight_layout()
    plt.savefig('animations/depth_temperature_profiles.png', dpi=150)
    plt.show()