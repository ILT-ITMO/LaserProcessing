import numpy as np
import matplotlib.pyplot as plt
import config
from matplotlib.animation import FuncAnimation, PillowWriter
from conditions import convert_to_physical_temperature, get_physical_extent
from matplotlib.patches import Circle, Polygon, Ellipse

# Константы для визуализации
TEMP_COLORMAP = 'hot'
ISOTHERM_TEMP = 1900
TARGET_DEPTHS = [10, 20, 30]
DEPTH_COLORS = ['cyan', 'lime', 'yellow']
FIG_SIZE_ANIM = (25, 14)
FIG_SIZE_LARGE = (20, 12)
FIG_SIZE_MEDIUM = (15, 8)
DPI = 150

# ============================================================================
# ЛАЗЕРНЫЙ КРАТЕР (НОВАЯ ВЕРСИЯ)
# ============================================================================

def laser_crater_3d(x_phys_um, y_phys_um, z_phys_um,
                    max_depth_um=None, crater_width_um=None,
                    max_height_um=None, decay_length_um=None):
    """
    Моделирует кратер от лазерного импульса с плавным затуханием в обе стороны.
    
    Физика: лазер создает кратер, который плавно затухает как вглубь материала (z<0),
    так и над поверхностью (z>0). Кратер достигает нуля точно на z = max_height_um.
    
    Параметры:
    -----------
    max_depth_um : float
        Максимальная глубина кратера в центре на поверхности (мкм)
    crater_width_um : float
        Полная ширина кратера на уровне 1% от максимума (мкм)
    max_height_um : float
        Высота, на которой кратер полностью затухает (z = max_height_um)
        Если None, то равна max_depth_um
    """
    if max_depth_um is None:
        max_depth_um = getattr(config, "CRATER_PEAK_DEPTH_UM", 30.0)
    if crater_width_um is None:
        crater_width_um = getattr(config, "CRATER_WIDTH_99_UM", 145.0)
    if max_height_um is None:
        max_height_um = max_depth_um

    # Расчет sigma для гауссова профиля по горизонтали
    R = crater_width_um / 2.0
    sigma_um = R / np.sqrt(-2 * np.log(0.01))  # = R/3.034
    
    print(f"Лазерный кратер с плавным затуханием до z={max_height_um}:")
    print(f"  Макс. глубина: {max_depth_um} мкм")
    print(f"  Ширина на поверхности: {crater_width_um} мкм")
    print(f"  Высота затухания: {max_height_um} мкм")
    print(f"  sigma = {sigma_um:.1f} мкм")
    
    X, Y, Z = np.meshgrid(x_phys_um, y_phys_um, z_phys_um, indexing='ij')
    
    # Гауссова форма в горизонтальной плоскости
    r_squared = X**2 + Y**2
    horizontal = np.exp(-r_squared / (2.0 * sigma_um**2))
    
    # Вертикальное затухание с плавным достижением нуля на z = max_height_um
    vertical = np.zeros_like(Z)
    
    # Для z < 0 (вглубь материала) - экспоненциальное затухание
    mask_down = Z < 0
    vertical[mask_down] = np.exp(Z[mask_down] / (max_depth_um / 2))
    
    # Для 0 <= z <= max_height_um - КВАДРАТИЧНОЕ затухание до нуля
    # Используем (1 - z/max_height_um)^2 для более плавного затухания
    mask_up = (Z >= 0) & (Z <= max_height_um)
    vertical[mask_up] = (1.0 - Z[mask_up] / max_height_um) ** 2
    
    # Для z = 0
    vertical[Z == 0] = 1.0
    
    # Для z > max_height_um - ноль (уже задано при инициализации)
    
    crater = max_depth_um * horizontal * vertical
    
    # Проверка
    if np.any(Z > 0):
        z_max_idx = np.argmin(np.abs(Z - max_height_um))
        val_at_max = np.max(crater[:, :, z_max_idx])
        print(f"  Значение на z={max_height_um:.1f}: {val_at_max:.6f} мкм")
        print(f"  Макс. значение при z>0: {np.max(crater[Z>0]):.3f} мкм")
    
    return crater


# ============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================================

def _add_crosshair(ax, color='white', alpha=0.5):
    """Добавляет перекрестие в центре графика."""
    ax.axhline(y=0, color=color, linestyle='--', alpha=alpha, linewidth=1)
    ax.axvline(x=0, color=color, linestyle='--', alpha=alpha, linewidth=1)
    ax.plot(0, 0, 'w+', markersize=10, markeredgewidth=1.5)

def _add_beam_circle(ax, beam_radius_um, color='cyan', alpha=0.7):
    """Добавляет круг радиуса пучка."""
    circle = Circle((0, 0), beam_radius_um, fill=False, color=color,
                    linestyle='-', linewidth=1.5, alpha=alpha)
    ax.add_patch(circle)

def _get_center_indices(x, y):
    """Возвращает центральные индексы для массивов x и y."""
    return len(x) // 2, len(y) // 2

def _get_mode_text():
    """Возвращает текст режима и мощности."""
    mode = "ИМПУЛЬСНЫЙ" if config.LASER_MODE == "pulsed" else "НЕПРЕРЫВНЫЙ"
    power = (f"{config.LASER_AVG_POWER} Вт (ср.)" if config.LASER_MODE == "pulsed" 
             else f"{config.LASER_CONTINUOUS_POWER} Вт")
    return mode, power

def _get_physical_coords(x_plot, y_plot, z_plot):
    """Преобразует координаты для визуализации."""
    (x_min, x_max), (y_min, y_max), (z_min, z_max) = get_physical_extent(
        [x_plot[0], x_plot[-1]], [y_plot[0], y_plot[-1]], [z_plot[0], z_plot[-1]]
    )
    x_phys = np.linspace(x_min, x_max, len(x_plot))
    y_phys = np.linspace(y_min, y_max, len(y_plot))
    z_phys = np.linspace(z_min, z_max, len(z_plot))
    return x_phys, y_phys, z_phys, (x_min, x_max), (y_min, y_max), (z_min, z_max)


# ============================================================================
# ВИЗУАЛИЗАЦИЯ ЛАЗЕРНЫХ ПРОФИЛЕЙ
# ============================================================================

def visualize_laser_pulses():
    """Визуализация временного профиля лазерных импульсов."""
    current_mode = config.LASER_MODE
    t_phys = None
    
    plt.figure(figsize=(12, 5))
    
    if current_mode == "pulsed":
        num_impulses = min(config.NUM_PULSES, 5)
        t_test = np.linspace(0, config.LASER_PULSE_PERIOD_NORM * num_impulses, 2000)
        
        t_mod = t_test % config.LASER_PULSE_PERIOD_NORM
        source_values = config.LASER_AMPLITUDE * np.exp(
            -(t_mod - config.LASER_PULSE_PERIOD_NORM/2)**2 / 
            (2 * config.LASER_PULSE_SIGMA_NORM**2)
        )
        
        t_phys = t_test * config.CHARACTERISTIC_TIME * 1e6
        plt.plot(t_phys, source_values, 'r-', linewidth=2.5, label='Интенсивность лазера')
        
        for i in range(num_impulses + 1):
            period_time = i * config.LASER_PULSE_PERIOD_NORM * config.CHARACTERISTIC_TIME * 1e6
            plt.axvline(x=period_time, color='gray', linestyle='--', alpha=0.5)
            if i < num_impulses:
                plt.text(period_time + 0.5, 0.85 * config.LASER_AMPLITUDE, 
                        f'Импульс {i+1}', fontsize=10, ha='left')
        
        first_end = config.LASER_PULSE_PERIOD_NORM * config.CHARACTERISTIC_TIME * 1e6
        plt.fill_between(t_phys[t_phys <= first_end], 0, source_values[t_phys <= first_end], 
                        alpha=0.3, color='red')
        
        title = (f'Временной профиль лазерных импульсов (Импульсный режим)\n'
                 f'Пиковая мощность: {config.LASER_PEAK_POWER:.1f} Вт, '
                 f'Частота: {config.LASER_REP_RATE:.0f} Гц, '
                 f'Длительность: {config.LASER_PULSE_DURATION*1e6:.1f} мкс')
        
        info = (f"Всего импульсов: {config.NUM_PULSES}\n"
                f"Период: {config.LASER_PULSE_PERIOD*1e6:.1f} мкс\n"
                f"Скважность: {config.LASER_DUTY_CYCLE*100:.1f}%\n"
                f"Средняя мощность: {config.LASER_AVG_POWER} Вт")
        color, bg = 'red', 'lightyellow'
        
    else:
        t_test = np.linspace(0, config.SIMULATION_TIME_NORM, 1000)
        source_values = np.ones_like(t_test) * config.LASER_AMPLITUDE
        
        t_phys = t_test * config.CHARACTERISTIC_TIME * 1e6
        plt.plot(t_phys, source_values, 'b-', linewidth=2.5, label='Интенсивность лазера')
        plt.fill_between(t_phys, 0, source_values, alpha=0.3, color='blue')
        plt.ylim(0, config.LASER_AMPLITUDE * 1.1)
        
        plt.axvline(x=0, color='green', linestyle='-', alpha=0.7, linewidth=2, label='Начало')
        plt.axvline(x=config.SIMULATION_TIME_PHYSICAL*1e6, color='red', linestyle='-', 
                   alpha=0.7, linewidth=2, label='Конец')
        
        title = (f'Временной профиль лазерного излучения (Непрерывный режим)\n'
                 f'Мощность: {config.LASER_CONTINUOUS_POWER} Вт, '
                 f'Длина волны: {config.LASER_WAVELENGTH*1e6:.1f} мкм')
        
        info = (f"Режим: НЕПРЕРЫВНЫЙ\n"
                f"Мощность: {config.LASER_CONTINUOUS_POWER} Вт\n"
                f"Интенсивность: {config.LASER_PEAK_INTENSITY/1e6:.2f} МВт/м²\n"
                f"Время моделирования: {config.SIMULATION_TIME_PHYSICAL*1e6:.1f} мкс")
        color, bg = 'blue', 'lightblue'
    
    plt.xlabel('Время (мкс)', fontsize=12)
    plt.ylabel('Относительная интенсивность', fontsize=12)
    plt.title(title, fontsize=13)
    plt.grid(True, alpha=0.3)
    
    plt.text(0.02, 0.98, info, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle="round,pad=0.3", facecolor=bg, alpha=0.8))
    
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(f'animations/laser_{current_mode}_profile.png', dpi=DPI, bbox_inches='tight')
    plt.show()


def visualize_laser_spatial_profile():
    """Визуализация пространственного профиля лазерного пучка."""
    current_mode = config.LASER_MODE
    
    x_test = np.linspace(-1, 1, 200)
    y_test = np.linspace(-1, 1, 200)
    X, Y = np.meshgrid(x_test, y_test)
    
    spatial_dist = config.LASER_AMPLITUDE * np.exp(-(X**2 + Y**2) / (config.LASER_SIGMA_NORM**2))
    X_phys = X * config.CHARACTERISTIC_LENGTH * 1e6
    Y_phys = Y * config.CHARACTERISTIC_LENGTH * 1e6
    beam_radius_um = config.LASER_BEAM_RADIUS * 1e6
    
    fig = plt.figure(figsize=(15, 6))
    
    # 2D контур
    ax1 = fig.add_subplot(1, 2, 1)
    contour = ax1.contourf(X_phys, Y_phys, spatial_dist, levels=50, cmap=TEMP_COLORMAP)
    plt.colorbar(contour, ax=ax1, label='Относительная интенсивность')
    ax1.set_xlabel('x (мкм)')
    ax1.set_ylabel('y (мкм)')
    
    _add_crosshair(ax1)
    _add_beam_circle(ax1, beam_radius_um, 'cyan', 0.8)
    
    radius_1e2 = beam_radius_um / np.sqrt(2)
    circle_1e2 = Circle((0, 0), radius_1e2, fill=False, color='lime',
                        linestyle='--', linewidth=1.5, alpha=0.7,
                        label=f'1/e² радиус: {radius_1e2:.0f} мкм')
    ax1.add_patch(circle_1e2)
    
    mode_text, power_text = _get_mode_text()
    ax1.set_title(f'Пространственное распределение лазерного пучка\nРежим: {mode_text}', fontsize=13)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.2)
    ax1.set_aspect('equal')
    
    # 3D поверхность
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    surf = ax2.plot_surface(X_phys[::4, ::4], Y_phys[::4, ::4], spatial_dist[::4, ::4],
                           cmap=TEMP_COLORMAP, alpha=0.8, rstride=1, cstride=1)
    ax2.set_xlabel('x (мкм)')
    ax2.set_ylabel('y (мкм)')
    ax2.set_zlabel('Интенсивность')
    ax2.set_title('3D профиль интенсивности', fontsize=13)
    ax2.view_init(elev=30, azim=45)
    
    info = f"Режим: {mode_text}\nМощность: {power_text}\nРадиус: {beam_radius_um:.0f} мкм"
    ax2.text2D(0.05, 0.95, info, transform=ax2.transAxes, fontsize=10, va='top',
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
    
    plt.suptitle('Характеристики лазерного пучка', fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig(f'animations/laser_spatial_profile_{current_mode}.png', dpi=DPI, bbox_inches='tight')
    plt.show()


# ============================================================================
# ИЗОТЕРМЫ
# ============================================================================

def add_isotherm_plot(ax, U_physical, x_phys, y_phys, z_phys, time_idx,
                      isotherm_temp=1900, mode_text="", power_text="",
                      beam_radius_um=None):
    """Добавляет график изотермы."""
    center_x, center_y = _get_center_indices(x_phys, y_phys)
    X_surf, Y_surf = np.meshgrid(x_phys, y_phys)
    temp_surface = U_physical[:, :, -1, time_idx]
    
    if np.max(temp_surface) >= isotherm_temp:
        cs = ax.contour(X_surf, Y_surf, temp_surface.T, levels=[isotherm_temp],
                       colors='white', linewidths=2.5, alpha=0.9)
        
        for path in cs.get_paths():
            vertices = path.vertices
            if len(vertices) > 2:
                polygon = Polygon(vertices, closed=True, fill=True,
                                 facecolor='lime', alpha=0.3, edgecolor='lime')
                ax.add_patch(polygon)
                
                if len(vertices) > 10:
                    mid = vertices[len(vertices)//2]
                    ax.text(mid[0], mid[1], f'{isotherm_temp} K', color='lime',
                           fontsize=9, fontweight='bold', ha='center', va='center',
                           bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.5))
    
    # Поиск глубины и ширины изотермы
    temp_profile = U_physical[center_x, center_y, :, time_idx]
    depth_at_isotherm = next((z_phys[i] for i in range(len(z_phys)-1, -1, -1) 
                             if temp_profile[i] >= isotherm_temp), None)
    
    width_at_isotherm = None
    if np.max(temp_surface) >= isotherm_temp:
        for i in range(center_x, len(x_phys)):
            if temp_surface[i, center_y] < isotherm_temp:
                width_at_isotherm = abs(x_phys[i])
                break
    
    info = f"ИЗОТЕРМА {isotherm_temp} K:\n"
    info += f"Глубина: {depth_at_isotherm:.1f} мкм\n" if depth_at_isotherm else "Глубина: не достигнута\n"
    info += f"Ширина: {width_at_isotherm:.1f} мкм" if width_at_isotherm else "Ширина: не достигнута"
    
    ax.text(0.02, 0.98, info, transform=ax.transAxes, fontsize=10, va='top',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7, edgecolor='lime'))
    
    _add_crosshair(ax)
    if beam_radius_um:
        _add_beam_circle(ax, beam_radius_um)
    
    ax.set_title(f'Изотерма {isotherm_temp} K на поверхности', fontsize=11)
    ax.set_xlabel('x (мкм)')
    ax.set_ylabel('y (мкм)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)


# ============================================================================
# АНИМАЦИЯ
# ============================================================================

def create_animation(U_data, x_plot, y_plot, z_plot, t_plot, title, filename, *, metrics=None):
    """Создает анимацию распределения температуры."""
    U_physical = convert_to_physical_temperature(U_data)
    x_phys, y_phys, z_phys, (x_min, x_max), (y_min, y_max), (z_min, z_max) = _get_physical_coords(x_plot, y_plot, z_plot)
    
    t_phys = t_plot * config.CHARACTERISTIC_TIME * 1e6
    z_vals = np.array(z_plot) * config.CHARACTERISTIC_LENGTH * 1e6
    
    depth_indices = [np.argmin(np.abs(z_vals - d)) for d in TARGET_DEPTHS]
    actual_depths = [z_vals[i] for i in depth_indices]
    
    # Подготовка кратера (НОВАЯ ВЕРСИЯ)
    crater_field = laser_crater_3d(
        x_phys, y_phys, z_phys,
        max_depth_um=getattr(config, "CRATER_PEAK_DEPTH_UM", 30.0),
        crater_width_um=getattr(config, "CRATER_WIDTH_99_UM", 145.0),
        decay_length_um=getattr(config, "CRATER_DECAY_LENGTH_UM", 30.0)  # добавить в config если нужно
    )
    
    surf_z_idx = np.argmin(np.abs(z_phys - 0.0))
    crater_surface = crater_field[:, :, surf_z_idx]  # теперь положительные значения
    crater_peak = getattr(config, "CRATER_PEAK_DEPTH_UM", 30.0)
    
    # Уровень для контура кратера (1% от максимума)
    level_99 = 0.01 * crater_peak
    
    X_surf, Y_surf = np.meshgrid(x_phys, y_phys)
    beam_radius_um = config.LASER_BEAM_RADIUS * 1e6
    center_x, center_y = len(x_plot) // 2, len(y_plot) // 2
    
    fig = plt.figure(figsize=FIG_SIZE_ANIM)
    metrics_text = None
    if isinstance(metrics, dict) and ("mse" in metrics or "mae" in metrics):
        try:
            mse_val = metrics.get("mse")
            mae_val = metrics.get("mae")
            parts = []
            if mse_val is not None:
                parts.append(f"MSE: {float(mse_val):.3g}")
            if mae_val is not None:
                parts.append(f"MAE: {float(mae_val):.3g}")
            if parts:
                metrics_text = "МЕТРИКИ:\n" + "\n".join(parts) + "\n"
        except Exception:
            metrics_text = None
    
    def update(frame):
        fig.clear()
        t_cur = t_phys[frame]
        
        # XY поверхность
        ax1 = fig.add_subplot(3, 5, 1)
        data_xy = U_physical[:, :, -1, frame].T
        im1 = ax1.imshow(data_xy, extent=[x_min, x_max, y_min, y_max],
                        origin='lower', cmap=TEMP_COLORMAP,
                        vmin=config.INITIAL_TEMPERATURE, vmax=np.max(U_physical))
        ax1.set_title('XY срез (поверхность)')
        ax1.set_xlabel('x (мкм)')
        ax1.set_ylabel('y (мкм)')
        _add_crosshair(ax1)
        _add_beam_circle(ax1, beam_radius_um)
        
        if np.max(data_xy) >= ISOTHERM_TEMP:
            ax1.contour(X_surf, Y_surf, data_xy, levels=[ISOTHERM_TEMP], colors='lime', linewidths=2)
        ax1.contour(X_surf, Y_surf, crater_surface.T, levels=[level_99], colors='red', 
                   linewidths=2, linestyles='--', alpha=0.9)
        plt.colorbar(im1, ax=ax1, shrink=0.8, label='Температура (K)')
        
        # XZ срез
        ax2 = fig.add_subplot(3, 5, 2)
        data_xz = U_physical[:, len(y_plot)//2, :, frame].T
        im2 = ax2.imshow(data_xz, extent=[x_min, x_max, z_min, z_max],
                        origin='lower', cmap=TEMP_COLORMAP,
                        vmin=config.INITIAL_TEMPERATURE, vmax=np.max(U_physical))
        ax2.set_title('XZ срез (центральный)')
        ax2.set_xlabel('x (мкм)')
        ax2.set_ylabel('z (мкм)')
        
        for depth, color in zip(actual_depths, DEPTH_COLORS):
            ax2.axhline(y=depth, color=color, linestyle='--', alpha=0.7)
        ax2.axvline(x=0, color='white', linestyle='--', alpha=0.5)
        
        if np.max(data_xz) >= ISOTHERM_TEMP:
            ax2.contour(data_xz, levels=[ISOTHERM_TEMP], colors='lime', linewidths=2,
                       extent=[x_min, x_max, z_min, z_max])
        
        crater_xz = crater_field[:, len(y_plot)//2, :]
        ax2.contour(x_phys, z_phys, crater_xz.T, levels=[level_99], colors='red',
                   linewidths=2, linestyles='--', alpha=0.9)
        plt.colorbar(im2, ax=ax2, shrink=0.8, label='Температура (K)')
        
        # YZ срез
        ax3 = fig.add_subplot(3, 5, 3)
        data_yz = U_physical[len(x_plot)//2, :, :, frame].T
        im3 = ax3.imshow(data_yz, extent=[y_min, y_max, z_min, z_max],
                        origin='lower', cmap=TEMP_COLORMAP,
                        vmin=config.INITIAL_TEMPERATURE, vmax=np.max(U_physical))
        ax3.set_title('YZ срез (центральный)')
        ax3.set_xlabel('y (мкм)')
        ax3.set_ylabel('z (мкм)')
        
        for depth, color in zip(actual_depths, DEPTH_COLORS):
            ax3.axhline(y=depth, color=color, linestyle='--', alpha=0.7)
        ax3.axvline(x=0, color='white', linestyle='--', alpha=0.5)
        
        if np.max(data_yz) >= ISOTHERM_TEMP:
            ax3.contour(data_yz, levels=[ISOTHERM_TEMP], colors='lime', linewidths=2,
                       extent=[y_min, y_max, z_min, z_max])
        
        crater_yz = crater_field[len(x_plot)//2, :, :]
        ax3.contour(y_phys, z_phys, crater_yz.T, levels=[level_99], colors='red',
                   linewidths=2, linestyles='--', alpha=0.9)
        plt.colorbar(im3, ax=ax3, shrink=0.8, label='Температура (K)')
        
        # XY срезы на глубинах
        for i, (d_idx, depth, color) in enumerate(zip(depth_indices, actual_depths, DEPTH_COLORS)):
            ax = fig.add_subplot(3, 5, 4 + i)
            data_depth = U_physical[:, :, d_idx, frame].T
            
            im = ax.imshow(data_depth, extent=[x_min, x_max, y_min, y_max],
                          origin='lower', cmap=TEMP_COLORMAP,
                          vmin=config.INITIAL_TEMPERATURE, vmax=np.max(U_physical))
            ax.set_title(f'XY срез (z = {depth:.1f} мкм)')
            ax.set_xlabel('x (мкм)')
            ax.set_ylabel('y (мкм)')
            
            _add_crosshair(ax)
            for spine in ax.spines.values():
                spine.set_color(color)
                spine.set_linewidth(2)
            
            if np.max(data_depth) >= ISOTHERM_TEMP:
                ax.contour(X_surf, Y_surf, data_depth, levels=[ISOTHERM_TEMP], colors='lime', linewidths=2)
            
            crater_slice = crater_field[:, :, d_idx]
            ax.contour(X_surf, Y_surf, crater_slice.T, levels=[level_99], colors='red',
                      linewidths=2, linestyles='--', alpha=0.9)
            plt.colorbar(im, ax=ax, shrink=0.8, label='Температура (K)')
        
        # График с изотермой
        ax7 = fig.add_subplot(3, 5, 7)
        mode_text, power_text = _get_mode_text()
        add_isotherm_plot(ax7, U_physical, x_phys, y_phys, z_phys, frame,
                         isotherm_temp=ISOTHERM_TEMP, mode_text=mode_text,
                         power_text=power_text, beam_radius_um=beam_radius_um)
        ax7.contour(X_surf, Y_surf, crater_surface.T, levels=[level_99], colors='red',
                   linewidths=2, linestyles='--', alpha=0.9)
        
        # Временной профиль лазера
        ax8 = fig.add_subplot(3, 5, 8)
        t_range = np.linspace(0, config.SIMULATION_TIME_NORM, 1000)
        t_range_phys = t_range * config.CHARACTERISTIC_TIME * 1e6
        
        if config.LASER_MODE == "pulsed":
            t_mod = t_range % config.LASER_PULSE_PERIOD_NORM
            laser_profile = config.LASER_AMPLITUDE * np.exp(
                -(t_mod - config.LASER_PULSE_PERIOD_NORM/2)**2 / (2 * config.LASER_PULSE_SIGMA_NORM**2))
            ax8.plot(t_range_phys, laser_profile, 'r-', linewidth=1.5)
            ax8.set_ylim(0, 1.1)
            ax8.set_title(f'Лазерные импульсы ({config.NUM_PULSES} импульсов)')
            
            for i in range(config.NUM_PULSES + 1):
                t_imp = i * config.LASER_PULSE_PERIOD_NORM * config.CHARACTERISTIC_TIME * 1e6
                ax8.axvline(x=t_imp, color='gray', linestyle=':', alpha=0.5)
        else:
            laser_profile = np.ones_like(t_range) * config.LASER_AMPLITUDE
            ax8.plot(t_range_phys, laser_profile, 'b-', linewidth=2)
            ax8.fill_between(t_range_phys, 0, laser_profile, alpha=0.3, color='blue')
            ax8.set_ylim(0, config.LASER_AMPLITUDE * 1.1)
            ax8.set_title('Непрерывный лазерный источник')
        
        ax8.axvline(x=t_cur, color='blue', linestyle='--', alpha=0.8, linewidth=2)
        ax8.set_xlabel('Время (мкс)')
        ax8.set_ylabel('Интенсивность')
        ax8.grid(True, alpha=0.3)
        ax8.set_xlim(0, max(t_phys))
        
        # Информация
        ax9 = fig.add_subplot(3, 5, 9)
        ax9.axis('off')
        
        if config.LASER_MODE == "pulsed":
            pulse_num = min(int(t_plot[frame] // config.LASER_PULSE_PERIOD_NORM) + 1, config.NUM_PULSES)
            t_in_pulse = (t_plot[frame] % config.LASER_PULSE_PERIOD_NORM) * config.CHARACTERISTIC_TIME * 1e6
            info = f"Импульс: {pulse_num}/{config.NUM_PULSES}\nВ импульсе: {t_in_pulse:.1f} мкс\n"
        else:
            info = f"Режим: НЕПРЕРЫВНЫЙ\nМощность: {config.LASER_CONTINUOUS_POWER} Вт\n"
        
        max_temp = np.max(U_physical[:, :, :, frame])
        min_temp = np.min(U_physical[:, :, :, frame])
        isotherm = "ДА" if max_temp >= ISOTHERM_TEMP else "НЕТ"
        
        info += f"\nТЕМПЕРАТУРА:\nМакс: {max_temp:.1f} K\nМин: {min_temp:.1f} K\n"
        info += f"Перегрев: {max_temp - config.INITIAL_TEMPERATURE:.1f} K\n"
        info += f"ИЗОТЕРМА {ISOTHERM_TEMP} K: {isotherm}\n\nНА ГЛУБИНАХ:\n"
        
        for depth_idx, depth in zip(depth_indices, actual_depths):
            temp = np.max(U_physical[:, :, depth_idx, frame])
            info += f"  {depth:.1f} мкм: {temp:.1f} K\n"
        
        ax9.text(0.1, 0.5, f"ВРЕМЯ: {t_cur:.1f} мкс\n" + "="*30 + "\n" + info,
                fontsize=10, va='center', linespacing=1.4,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        # Пространственный профиль
        ax10 = fig.add_subplot(3, 5, 11)
        x_prof = np.linspace(-1, 1, 100) * config.CHARACTERISTIC_LENGTH * 1e6
        laser_spatial = config.LASER_AMPLITUDE * np.exp(-np.linspace(-1, 1, 100)**2 / config.LASER_SIGMA_NORM**2)
        ax10.plot(x_prof, laser_spatial, 'g-', linewidth=2.5)
        ax10.set_xlabel('x (мкм)')
        ax10.set_ylabel('Интенсивность')
        ax10.set_title(f'Профиль пучка (радиус {beam_radius_um:.0f} мкм)')
        ax10.grid(True, alpha=0.3)
        ax10.axvline(x=beam_radius_um, color='cyan', linestyle='--', alpha=0.7)
        ax10.axvline(x=-beam_radius_um, color='cyan', linestyle='--', alpha=0.7)
        
        # Температура по глубине
        ax11 = fig.add_subplot(3, 5, 12)
        temp_depth = U_physical[center_x, center_y, :, frame]
        ax11.plot(temp_depth, z_vals, 'b-', linewidth=2.5)
        ax11.set_xlabel('Температура (K)')
        ax11.set_ylabel('Глубина z (мкм)')
        ax11.set_title('Температура по глубине\n(в центре пучка)')
        ax11.grid(True, alpha=0.3)
        ax11.axvline(x=ISOTHERM_TEMP, color='lime', linestyle='-', alpha=0.5, label=f'Изотерма {ISOTHERM_TEMP} K')
        
        for depth, color in zip(actual_depths, DEPTH_COLORS):
            d_idx = np.argmin(np.abs(z_vals - depth))
            t = temp_depth[d_idx]
            ax11.plot(t, depth, 'o', color=color, markersize=8, markeredgecolor='black')
            ax11.text(t + 5, depth, f'{t:.0f}K', fontsize=9, va='center')
        ax11.legend(fontsize=9)
        
        # Профиль вдоль X
        ax12 = fig.add_subplot(3, 5, 13)
        x_profile = U_physical[:, len(y_plot)//2, -1, frame]
        ax12.plot(x_phys, x_profile, 'r-', linewidth=2.5, label='Поверхность')
        ax12.fill_between(x_phys, config.INITIAL_TEMPERATURE, x_profile, alpha=0.2, color='red')
        ax12.axhline(y=ISOTHERM_TEMP, color='lime', linestyle='-', alpha=0.7, linewidth=2,
                    label=f'Изотерма {ISOTHERM_TEMP} K')
        
        crater_line = crater_surface[crater_surface.shape[0]//2, :]
        # Для кратера теперь положительные значения, не нужно брать минус
        crater_temp = ISOTHERM_TEMP - crater_peak + crater_line
        ax12.plot(x_phys, crater_temp, 'r--', linewidth=2.0, label='Лазерный кратер')
        
        ax12.set_xlabel('x (мкм)')
        ax12.set_ylabel('Температура (K)')
        ax12.set_title('Профиль температуры вдоль X\n(поверхность, y=0)')
        ax12.grid(True, alpha=0.3)
        ax12.legend(fontsize=9)
        
        # Зона нагрева
        ax13 = fig.add_subplot(3, 5, 14)
        surface_temp = U_physical[:, center_y, -1, frame]
        left = next((i for i in range(center_x, len(x_phys)) if surface_temp[i] >= ISOTHERM_TEMP), None)
        right = next((i for i in range(center_x, len(x_phys)) if surface_temp[i] < ISOTHERM_TEMP), None)
        
        width = x_phys[right] - x_phys[left] if left and right else None
        depth = next((z_phys[i] for i in range(len(z_phys)) 
                     if U_physical[center_x, center_y, i, frame] >= ISOTHERM_TEMP), None)
        
        if width and depth:
            ellipse = Ellipse((0, -depth/2), width=width, height=depth,
                             edgecolor='lime', facecolor='lime', alpha=0.3, linewidth=2,
                             label=f'Зона >{ISOTHERM_TEMP} K')
            ax13.add_patch(ellipse)
            ax13.text(0, -depth/2, f'Ш: {width:.1f} мкм\nГ: {depth:.1f} мкм',
                     ha='center', va='center', fontsize=10, fontweight='bold',
                     bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
        
        ax13.set_xlim(-beam_radius_um*2, beam_radius_um*2)
        ax13.set_ylim(-z_max, 0)
        ax13.set_xlabel('Ширина (мкм)')
        ax13.set_ylabel('Глубина (мкм)')
        ax13.set_title(f'Зона нагрева >{ISOTHERM_TEMP} K')
        ax13.grid(True, alpha=0.3)
        ax13.set_aspect('equal')
        ax13.invert_yaxis()
        
        crater_center = crater_field[center_x, :, :]
        ax13.contour(y_phys, z_phys, crater_center.T, levels=[level_99], colors='red',
                    linewidths=2, linestyles='--', alpha=0.9)
        
        # Итоговая информация
        ax15 = fig.add_subplot(3, 5, 15)
        ax15.axis('off')
        
        summary = f"СВОДКА ПО ИЗОТЕРМЕ {ISOTHERM_TEMP} K\n" + "="*25 + "\n\n"
        summary += f"• Ширина зоны: {width:.1f} мкм\n" if width else "• Ширина зоны: не достигнута\n"
        summary += f"• Глубина: {depth:.1f} мкм\n" if depth else "• Глубина: не достигнута\n"
        summary += f"• Макс. температура: {max_temp:.1f} K\n"
        summary += f"• Перегрев: {max_temp - config.INITIAL_TEMPERATURE:.1f} K\n"
        if metrics_text:
            summary += "\n" + metrics_text
        
        if width and depth:
            summary += f"• Отношение Ш/Г: {width/depth:.2f}"
        
        ax15.text(0.1, 0.5, summary, fontsize=10, va='center', linespacing=1.5,
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
        
        mode_text = "ИМПУЛЬСНЫЙ" if config.LASER_MODE == "pulsed" else "НЕПРЕРЫВНЫЙ"
        pulse_info = f" (Импульс {pulse_num}/{config.NUM_PULSES})" if config.LASER_MODE == "pulsed" else ""
        metrics_line = ""
        if metrics_text:
            # Однострочная версия для заголовка
            try:
                mse_val = metrics.get("mse") if isinstance(metrics, dict) else None
                mae_val = metrics.get("mae") if isinstance(metrics, dict) else None
                bits = []
                if mse_val is not None:
                    bits.append(f"MSE={float(mse_val):.3g}")
                if mae_val is not None:
                    bits.append(f"MAE={float(mae_val):.3g}")
                if bits:
                    metrics_line = " | " + ", ".join(bits)
            except Exception:
                metrics_line = ""

        plt.suptitle(f'{title}\nРежим: {mode_text}, Время: {t_cur:.1f} мкс{pulse_info}{metrics_line}',
                    fontsize=14, y=0.98)
        plt.tight_layout()
        
        return fig,
    
    ani = FuncAnimation(fig, update, frames=len(t_plot), interval=500, blit=False, repeat=True)
    ani.save(filename, writer=PillowWriter(fps=2), dpi=100)
    plt.close(fig)
    print(f"Анимация сохранена: {filename}")
    return ani


# ============================================================================
# ЭВОЛЮЦИЯ ТЕМПЕРАТУРЫ
# ============================================================================

def plot_temperature_evolution(U_data, x_plot, y_plot, z_plot, t_plot):
    """График эволюции температуры в центре пучка."""
    U_physical = convert_to_physical_temperature(U_data)
    t_phys = t_plot * config.CHARACTERISTIC_TIME * 1e6
    
    center_x, center_y = len(x_plot)//2, len(y_plot)//2
    center_temp = U_physical[center_x, center_y, -1, :]
    
    plt.figure(figsize=(14, 7))
    plt.plot(t_phys, center_temp, 'b-', linewidth=3, label='Температура в центре пучка')
    plt.axhline(y=config.INITIAL_TEMPERATURE, color='gray', linestyle='--', linewidth=2,
                label=f'Начальная ({config.INITIAL_TEMPERATURE} K)')
    plt.axhline(y=ISOTHERM_TEMP, color='lime', linestyle='-', linewidth=2, alpha=0.7,
                label=f'Изотерма {ISOTHERM_TEMP} K')
    
    if config.LASER_MODE == "pulsed":
        colors = plt.cm.Reds(np.linspace(0.3, 0.8, config.NUM_PULSES))
        for i in range(config.NUM_PULSES):
            t_start = i * config.LASER_PULSE_PERIOD_NORM * config.CHARACTERISTIC_TIME * 1e6
            t_end = t_start + config.LASER_PULSE_DURATION_NORM * config.CHARACTERISTIC_TIME * 1e6
            plt.axvspan(t_start, t_end, alpha=0.15, color=colors[i])
            if i == 0:
                plt.axvline(x=t_end, color='red', linestyle=':', alpha=0.6, label='Границы импульсов')
        
        title = (f'Эволюция температуры в центре лазерного пучка\n'
                 f'Импульсный режим: {config.NUM_PULSES} импульсов, '
                 f'{config.LASER_AVG_POWER} Вт (ср.), {config.LASER_REP_RATE:.0f} Гц')
    else:
        plt.fill_between(t_phys, config.INITIAL_TEMPERATURE, center_temp, alpha=0.2, color='blue')
        title = (f'Эволюция температуры в центре лазерного пучка\n'
                 f'Непрерывный режим: {config.LASER_CONTINUOUS_POWER} Вт, '
                 f'Время моделирования: {config.SIMULATION_TIME_PHYSICAL*1e6:.1f} мкс')
    
    plt.xlabel('Время (мкс)')
    plt.ylabel('Температура (K)')
    plt.title(title, fontsize=13)
    plt.grid(True, alpha=0.3)
    
    max_temp = np.max(center_temp)
    max_time = t_phys[np.argmax(center_temp)]
    isotherm_reached = np.any(center_temp >= ISOTHERM_TEMP)
    
    info = (f"Макс. температура: {max_temp:.1f} K\n"
            f"Перегрев: {max_temp - config.INITIAL_TEMPERATURE:.1f} K\n"
            f"Время макс. нагрева: {max_time:.1f} мкс\n"
            f"Изотерма {ISOTHERM_TEMP} K: {'ДА' if isotherm_reached else 'НЕТ'}")
    
    if isotherm_reached:
        isotherm_time = t_phys[np.where(center_temp >= ISOTHERM_TEMP)[0][0]]
        info += f"\nВремя достижения: {isotherm_time:.1f} мкс"
        plt.scatter([isotherm_time], [ISOTHERM_TEMP], color='lime', s=100, zorder=5,
                   edgecolor='black', linewidth=2, label=f'Изотерма {ISOTHERM_TEMP} K')
    
    plt.text(0.02, 0.98, info, transform=plt.gca().transAxes, fontsize=11, va='top',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
    
    plt.scatter([max_time], [max_temp], color='red', s=100, zorder=5,
                label=f'Максимум: {max_temp:.1f} K')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(f'animations/temperature_evolution_center_{config.LASER_MODE}.png', dpi=DPI, bbox_inches='tight')
    plt.show()
    
    return center_temp


# ============================================================================
# ПРОФИЛИ ПО ГЛУБИНЕ
# ============================================================================

def plot_depth_temperature_profiles(U_data, x_plot, y_plot, z_plot, t_plot):
    """Графики распределения температуры по глубине в разные моменты."""
    U_physical = convert_to_physical_temperature(U_data)
    z_phys = np.array(z_plot) * config.CHARACTERISTIC_LENGTH * 1e6
    center_x, center_y = len(x_plot)//2, len(y_plot)//2
    
    if config.LASER_MODE == "pulsed":
        n_frames = min(config.NUM_PULSES, 8)
        key_frames = [min(int((i + 0.5) * len(t_plot) / n_frames), len(t_plot)-1) for i in range(n_frames)]
        title_suffix = 'во время импульсов'
    else:
        key_frames = np.linspace(0, len(t_plot)-1, min(8, len(t_plot)), dtype=int)
        title_suffix = 'в разные моменты времени'
    
    plt.figure(figsize=(12, 9))
    colors = plt.cm.viridis(np.linspace(0, 1, len(key_frames)))
    
    for i, f_idx in enumerate(key_frames):
        profile = U_physical[center_x, center_y, :, f_idx]
        t_val = t_plot[f_idx] * config.CHARACTERISTIC_TIME * 1e6
        
        if config.LASER_MODE == "pulsed":
            pulse = int(t_plot[f_idx] // config.LASER_PULSE_PERIOD_NORM) + 1
            label = f'Импульс {pulse} ({t_val:.0f} мкс)'
        else:
            label = f'{t_val:.0f} мкс'
        
        plt.plot(profile, z_phys, color=colors[i], linewidth=2.5, label=label)
    
    plt.axvline(x=ISOTHERM_TEMP, color='lime', linestyle='-', linewidth=2.5, alpha=0.7,
                label=f'Изотерма {ISOTHERM_TEMP} K')
    plt.xlabel('Температура (K)')
    plt.ylabel('Глубина (мкм)')
    
    mode_text = "Импульсный" if config.LASER_MODE == "pulsed" else "Непрерывный"
    power = (f"{config.LASER_AVG_POWER} Вт (ср.)" if config.LASER_MODE == "pulsed" 
             else f"{config.LASER_CONTINUOUS_POWER} Вт")
    
    plt.title(f'Распределение температуры по глубине {title_suffix}\nРежим: {mode_text}, Мощность: {power}', fontsize=13)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.gca().invert_yaxis()
    
    for depth in TARGET_DEPTHS:
        plt.axhline(y=depth, color='gray', linestyle=':', alpha=0.5)
        plt.text(config.INITIAL_TEMPERATURE + 10, depth, f'{depth} мкм', fontsize=9, color='gray', va='center')
    
    plt.tight_layout()
    plt.savefig(f'animations/depth_temperature_profiles_{config.LASER_MODE}.png', dpi=DPI, bbox_inches='tight')
    plt.show()


# ============================================================================
# СРАВНЕНИЕ РЕЖИМОВ
# ============================================================================

def plot_comparison_pulse_vs_continuous(temp_pulsed, temp_continuous, t_plot,
                                        pulsed_params=None, continuous_params=None):
    """Сравнение импульсного и непрерывного режимов."""
    if temp_pulsed is None or temp_continuous is None:
        print("Для сравнения нужны данные обоих режимов!")
        return
    
    t_phys = t_plot * config.CHARACTERISTIC_TIME * 1e6
    
    pulsed_params = pulsed_params or {"num_pulses": 8, "avg_power": 10.0, "rep_rate": 8000.0}
    continuous_params = continuous_params or {"power": 5.0}
    
    max_pulsed, max_cont = np.max(temp_pulsed), np.max(temp_continuous)
    time_pulsed = t_phys[np.argmax(temp_pulsed)]
    time_cont = t_phys[np.argmax(temp_continuous)]
    isotherm_pulsed = np.any(temp_pulsed >= ISOTHERM_TEMP)
    isotherm_cont = np.any(temp_continuous >= ISOTHERM_TEMP)
    
    plt.figure(figsize=(15, 8))
    
    line_pulsed = plt.plot(t_phys, temp_pulsed, 'r-', linewidth=3, alpha=0.8, label='Импульсный режим')[0]
    line_cont = plt.plot(t_phys, temp_continuous, 'b-', linewidth=3, alpha=0.8, label='Непрерывный режим')[0]
    
    plt.axhline(y=config.INITIAL_TEMPERATURE, color='gray', linestyle='--', linewidth=2, alpha=0.5,
                label=f'Начальная ({config.INITIAL_TEMPERATURE} K)')
    plt.axhline(y=ISOTHERM_TEMP, color='lime', linestyle='-', linewidth=2.5, alpha=0.8,
                label=f'Изотерма {ISOTHERM_TEMP} K')
    
    plt.fill_between(t_phys, temp_pulsed, temp_continuous,
                     where=temp_pulsed >= temp_continuous, color='red', alpha=0.15,
                     label='Импульсный > Непрерывный')
    plt.fill_between(t_phys, temp_pulsed, temp_continuous,
                     where=temp_pulsed < temp_continuous, color='blue', alpha=0.15,
                     label='Импульсный < Непрерывный')
    
    plt.scatter([time_pulsed], [max_pulsed], color='red', s=150, zorder=5, edgecolor='black',
                label=f'Макс. импульсный: {max_pulsed:.1f} K')
    plt.scatter([time_cont], [max_cont], color='blue', s=150, zorder=5, edgecolor='black',
                label=f'Макс. непрерывный: {max_cont:.1f} K')
    
    info = (f"СРАВНЕНИЕ РЕЖИМОВ\n" + "="*40 + "\n\n"
            f"ИМПУЛЬСНЫЙ:\n• Импульсов: {pulsed_params['num_pulses']}\n"
            f"• Мощность: {pulsed_params['avg_power']} Вт (ср.)\n"
            f"• Частота: {pulsed_params['rep_rate']:.0f} Гц\n"
            f"• Макс. темп.: {max_pulsed:.1f} K\n"
            f"• Изотерма {ISOTHERM_TEMP} K: {'ДА' if isotherm_pulsed else 'НЕТ'}\n\n"
            f"НЕПРЕРЫВНЫЙ:\n• Мощность: {continuous_params['power']} Вт\n"
            f"• Макс. темп.: {max_cont:.1f} K\n"
            f"• Изотерма {ISOTHERM_TEMP} K: {'ДА' if isotherm_cont else 'НЕТ'}\n\n"
            f"СРАВНЕНИЕ:\n• Разница: {abs(max_pulsed - max_cont):.1f} K\n"
            f"• {'Импульсный' if max_pulsed > max_cont else 'Непрерывный'} горячее на "
            f"{abs(max_pulsed/max_cont - 1) * 100:.1f}%")
    
    plt.text(0.02, 0.98, info, transform=plt.gca().transAxes, fontsize=10, va='top', linespacing=1.5,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9))
    
    plt.xlabel('Время (мкс)')
    plt.ylabel('Температура (K)')
    plt.title('Сравнение температурных профилей: Импульсный vs Непрерывный режим', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    from matplotlib.patches import Patch
    plt.legend(handles=[line_pulsed, line_cont,
                        Patch(facecolor='red', alpha=0.15, label='Импульсный > Непрерывный'),
                        Patch(facecolor='blue', alpha=0.15, label='Импульсный < Непрерывный')],
              loc='lower right')
    
    plt.tight_layout()
    plt.savefig('animations/comparison_pulse_vs_continuous.png', dpi=DPI, bbox_inches='tight')
    plt.show()


# ============================================================================
# АНАЛИЗ ТЕМПЕРАТУРНОГО ПОЛЯ
# ============================================================================

def plot_temperature_distribution_at_time(U_data, x_plot, y_plot, z_plot, t_plot, time_idx=-1):
    """Детальный анализ температурного поля в заданный момент времени."""
    U_physical = convert_to_physical_temperature(U_data)
    x_phys = np.array(x_plot) * config.CHARACTERISTIC_LENGTH * 1e6
    y_phys = np.array(y_plot) * config.CHARACTERISTIC_LENGTH * 1e6
    z_phys = np.array(z_plot) * config.CHARACTERISTIC_LENGTH * 1e6
    t_phys = t_plot[time_idx] * config.CHARACTERISTIC_TIME * 1e6
    
    X_surf, Y_surf = np.meshgrid(x_phys, y_phys)
    center_x, center_y = len(x_plot)//2, len(y_plot)//2
    
    # Кратер (НОВАЯ ВЕРСИЯ)
    crater_field = laser_crater_3d(
        x_phys, y_phys, z_phys,
        max_depth_um=getattr(config, "CRATER_PEAK_DEPTH_UM", 30.0),
        crater_width_um=getattr(config, "CRATER_WIDTH_99_UM", 145.0),
        decay_length_um=getattr(config, "CRATER_DECAY_LENGTH_UM", 30.0)
    )
    
    surf_z_idx = np.argmin(np.abs(z_phys - 0.0))
    crater_surface = crater_field[:, :, surf_z_idx].T
    crater_peak = getattr(config, "CRATER_PEAK_DEPTH_UM", 30.0)
    level_99 = 0.01 * crater_peak
    
    Z_surface = U_physical[:, :, -1, time_idx].T
    beam_radius_um = config.LASER_BEAM_RADIUS * 1e6
    
    fig = plt.figure(figsize=FIG_SIZE_LARGE)
    
    # 3D поверхность
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    X, Y = np.meshgrid(x_phys, y_phys)
    surf = ax1.plot_surface(X, Y, Z_surface, cmap=TEMP_COLORMAP, alpha=0.85,
                           rstride=1, cstride=1, linewidth=0.25)
    ax1.set_xlabel('x (мкм)')
    ax1.set_ylabel('y (мкм)')
    ax1.set_zlabel('Температура (K)')
    ax1.set_title(f'Температура на поверхности\nВремя: {t_phys:.1f} мкс')
    fig.colorbar(surf, ax=ax1, shrink=0.6, pad=0.1, label='Температура (K)')
    ax1.view_init(elev=30, azim=45)
    
    # Изотермы на поверхности
    ax2 = fig.add_subplot(2, 3, 2)
    contour = ax2.contourf(X, Y, Z_surface, levels=20, cmap=TEMP_COLORMAP)
    
    if np.max(Z_surface) >= ISOTHERM_TEMP:
        cs = ax2.contour(X, Y, Z_surface, levels=[ISOTHERM_TEMP], colors='lime',
                        linewidths=3, alpha=0.9)
        for path in cs.get_paths():
            if len(path.vertices) > 2:
                polygon = Polygon(path.vertices, closed=True, fill=True,
                                 facecolor='lime', alpha=0.3, edgecolor='lime', linewidth=2)
                ax2.add_patch(polygon)
    
    ax2.contour(X, Y, crater_surface, levels=[level_99], colors='red',
                linewidths=2, linestyles='--', alpha=0.9)
    ax2.contour(X, Y, Z_surface, levels=10, colors='white', linewidths=0.5, alpha=0.5)
    ax2.set_xlabel('x (мкм)')
    ax2.set_ylabel('y (мкм)')
    ax2.set_title('Изотермы на поверхности (1900 K выделена)')
    ax2.grid(True, alpha=0.3)
    _add_beam_circle(ax2, beam_radius_um)
    fig.colorbar(contour, ax=ax2, shrink=0.9, label='Температура (K)')
    ax2.set_aspect('equal')
    
    # Распределение по глубине
    ax3 = fig.add_subplot(2, 3, 3)
    temp_profile = U_physical[center_x, center_y, :, time_idx]
    ax3.plot(temp_profile, z_phys, 'b-', linewidth=3)
    ax3.axvline(x=ISOTHERM_TEMP, color='lime', linestyle='-', linewidth=2.5, alpha=0.7,
                label=f'Изотерма {ISOTHERM_TEMP} K')
    ax3.set_xlabel('Температура (K)')
    ax3.set_ylabel('Глубина z (мкм)')
    ax3.set_title('Распределение по глубине (центр пучка)')
    ax3.grid(True, alpha=0.3)
    ax3.invert_yaxis()
    
    for depth, color in zip(TARGET_DEPTHS, DEPTH_COLORS):
        d_idx = np.argmin(np.abs(z_phys - depth))
        t = temp_profile[d_idx]
        ax3.plot(t, depth, 'o', color=color, markersize=10, markeredgecolor='black')
        ax3.text(t + 5, depth, f'{t:.0f} K', fontsize=10, va='center',
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))
    ax3.legend(fontsize=10)
    
    # Срез на глубине 20 мкм
    ax4 = fig.add_subplot(2, 3, 4)
    depth_idx = np.argmin(np.abs(z_phys - 20))
    depth_slice = U_physical[:, :, depth_idx, time_idx].T
    
    im4 = ax4.imshow(depth_slice, extent=[x_phys[0], x_phys[-1], y_phys[0], y_phys[-1]],
                    origin='lower', cmap=TEMP_COLORMAP)
    ax4.set_xlabel('x (мкм)')
    ax4.set_ylabel('y (мкм)')
    ax4.set_title(f'Срез на глубине {z_phys[depth_idx]:.0f} мкм')
    ax4.grid(True, alpha=0.3)
    
    if np.max(depth_slice) >= ISOTHERM_TEMP:
        ax4.contour(X_surf, Y_surf, depth_slice, levels=[ISOTHERM_TEMP], colors='lime', linewidths=2)
    
    crater_slice = crater_field[:, :, depth_idx].T
    ax4.contour(X_surf, Y_surf, crater_slice, levels=[level_99], colors='red',
                linewidths=2, linestyles='--', alpha=0.9)
    _add_crosshair(ax4)
    fig.colorbar(im4, ax=ax4, shrink=0.9, label='Температура (K)')
    
    # Профиль вдоль X
    ax5 = fig.add_subplot(2, 3, 5)
    x_profile = U_physical[:, len(y_plot)//2, -1, time_idx]
    ax5.plot(x_phys, x_profile, 'r-', linewidth=3, label='Поверхность (y=0)')
    ax5.fill_between(x_phys, config.INITIAL_TEMPERATURE, x_profile, alpha=0.2, color='red')
    ax5.axhline(y=ISOTHERM_TEMP, color='lime', linestyle='-', linewidth=2.5, alpha=0.8,
                label=f'Изотерма {ISOTHERM_TEMP} K')
    
    center_idx = len(x_phys)//2
    left = next((i for i in range(center_idx, len(x_phys)) if x_profile[i] >= ISOTHERM_TEMP), None)
    right = next((i for i in range(center_idx, len(x_phys)) if x_profile[i] < ISOTHERM_TEMP), None)
    
    if left and right:
        ax5.fill_between(x_phys[left:right+1], ISOTHERM_TEMP, x_profile[left:right+1],
                        alpha=0.3, color='lime', label=f'Зона >{ISOTHERM_TEMP} K')
    
    ax5.set_xlabel('x (мкм)')
    ax5.set_ylabel('Температура (K)')
    ax5.set_title('Профиль температуры вдоль оси X')
    ax5.grid(True, alpha=0.3)
    ax5.axvline(x=beam_radius_um, color='cyan', linestyle='--', alpha=0.7, linewidth=2)
    ax5.axvline(x=-beam_radius_um, color='cyan', linestyle='--', alpha=0.7, linewidth=2)
    ax5.legend(fontsize=10)
    
    # Информационная панель
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    
    max_temp = np.max(U_physical[:, :, :, time_idx])
    isotherm = max_temp >= ISOTHERM_TEMP
    
    width = depth = None
    if isotherm:
        surface_temp = U_physical[:, center_y, -1, time_idx]
        left = next((i for i in range(center_x, len(x_phys)) if surface_temp[i] >= ISOTHERM_TEMP), None)
        right = next((i for i in range(center_x, len(x_phys)) if surface_temp[i] < ISOTHERM_TEMP), None)
        if left and right:
            width = x_phys[right] - x_phys[left]
        
        center_profile = U_physical[center_x, center_y, :, time_idx]
        depth = next((z_phys[i] for i in range(len(z_phys)) if center_profile[i] >= ISOTHERM_TEMP), None)
    
    info = (f"АНАЛИЗ ТЕМПЕРАТУРНОГО ПОЛЯ\n" + "="*40 + f"\n\nВРЕМЯ: {t_phys:.1f} мкс\n\n"
            f"ТЕМПЕРАТУРНЫЕ ХАРАКТЕРИСТИКИ:\n"
            f"• Максимальная: {max_temp:.1f} K\n"
            f"• Минимальная: {np.min(U_physical[:, :, :, time_idx]):.1f} K\n"
            f"• Перегрев: {max_temp - config.INITIAL_TEMPERATURE:.1f} K\n\n"
            f"ИЗОТЕРМА {ISOTHERM_TEMP} K:\n"
            f"• Достигнута: {'ДА' if isotherm else 'НЕТ'}\n")
    
    if width:
        info += f"• Ширина зоны: {width:.1f} мкм\n"
    if depth:
        info += f"• Глубина проникновения: {depth:.1f} мкм\n"
    
    mode_text, power_text = _get_mode_text()
    info += f"\nРЕЖИМ: {mode_text}\nМощность: {power_text}"
    
    ax6.text(0.1, 0.95, info, fontsize=11, va='top', linespacing=1.6,
             bbox=dict(boxstyle="round,pad=0.6", facecolor="lightblue", alpha=0.9))
    
    plt.suptitle(f'Детальный анализ температурного поля\nРежим: {mode_text}, Мощность: {power_text}, Время: {t_phys:.1f} мкс',
                fontsize=15, y=0.98)
    
    plt.tight_layout()
    plt.savefig(f'animations/temperature_field_analysis_{config.LASER_MODE}_t{t_phys:.0f}us.png',
               dpi=DPI, bbox_inches='tight')
    plt.show()


# ============================================================================
# ПРОЧИЕ ВИЗУАЛИЗАЦИИ
# ============================================================================

def plot_laser_intensity_3d():
    """Визуализация интенсивности лазерного пучка в 3D и 2D."""
    x = np.linspace(-1, 1, 100) * config.CHARACTERISTIC_LENGTH * 1e6
    y = np.linspace(-1, 1, 100) * config.CHARACTERISTIC_LENGTH * 1e6
    X, Y = np.meshgrid(x, y)
    
    intensity = np.exp(-(X**2 + Y**2) / (config.LASER_BEAM_RADIUS*1e6)**2)
    radius = config.LASER_BEAM_RADIUS * 1e6
    
    fig = plt.figure(figsize=FIG_SIZE_MEDIUM)
    
    # 3D
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    surf = ax1.plot_surface(X, Y, intensity, cmap=TEMP_COLORMAP, alpha=0.85,
                           rstride=2, cstride=2, linewidth=0.3, edgecolor='black')
    ax1.set_xlabel('x (мкм)')
    ax1.set_ylabel('y (мкм)')
    ax1.set_zlabel('Относительная интенсивность')
    ax1.set_title('3D профиль лазерного пучка')
    fig.colorbar(surf, ax=ax1, shrink=0.6, pad=0.1, label='Интенсивность')
    ax1.view_init(elev=35, azim=45)
    
    # 2D
    ax2 = fig.add_subplot(1, 2, 2)
    contour = ax2.contourf(X, Y, intensity, levels=30, cmap=TEMP_COLORMAP)
    ax2.contour(X, Y, intensity, levels=15, colors='white', linewidths=0.5, alpha=0.5)
    
    circles = [
        (radius, 'cyan', '-', 3, f'Радиус пучка: {radius:.0f} мкм'),
        (radius/np.sqrt(2), 'lime', '--', 2, f'1/e² радиус: {radius/np.sqrt(2):.0f} мкм'),
        (radius, 'yellow', ':', 2, f'1/e радиус: {radius:.0f} мкм')
    ]
    
    for r, color, ls, lw, label in circles:
        ax2.add_patch(Circle((0, 0), r, fill=False, color=color,
                            linestyle=ls, linewidth=lw, alpha=0.7, label=label))
    
    ax2.set_xlabel('x (мкм)')
    ax2.set_ylabel('y (мкм)')
    ax2.set_title('2D распределение интенсивности')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')
    ax2.set_aspect('equal')
    fig.colorbar(contour, ax=ax2, shrink=0.9, label='Интенсивность')
    
    mode_text, power_text = _get_mode_text()
    plt.suptitle(f'Лазерный пучок: {mode_text} режим\nМощность: {power_text}, '
                 f'λ={config.LASER_WAVELENGTH*1e6:.1f} мкм, Радиус: {radius:.0f} мкм',
                fontsize=14, y=0.95)
    
    plt.tight_layout()
    plt.savefig('animations/laser_intensity_3d_profile.png', dpi=DPI, bbox_inches='tight')
    plt.show()


def plot_heating_dynamics_comparison(U_data_pulsed, U_data_continuous, x_plot, y_plot, z_plot, t_plot):
    """Сравнение динамики нагрева в разных режимах."""
    U_pulsed = convert_to_physical_temperature(U_data_pulsed)
    U_cont = convert_to_physical_temperature(U_data_continuous)
    t_phys = t_plot * config.CHARACTERISTIC_TIME * 1e6
    
    center_x, center_y = len(x_plot)//2, len(y_plot)//2
    surface_z = len(z_plot) - 1
    depth_idx = np.argmin(np.abs(np.array(z_plot) * config.CHARACTERISTIC_LENGTH * 1e6 - 30))
    
    temp_center_pulsed = U_pulsed[center_x, center_y, surface_z, :]
    temp_center_cont = U_cont[center_x, center_y, surface_z, :]
    
    fig = plt.figure(figsize=(18, 12))
    
    # Температура в центре
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(t_phys, temp_center_pulsed, 'r-', linewidth=3, alpha=0.8, label='Импульсный')
    ax1.plot(t_phys, temp_center_cont, 'b-', linewidth=3, alpha=0.8, label='Непрерывный')
    ax1.axhline(y=config.INITIAL_TEMPERATURE, color='gray', linestyle='--', alpha=0.5, label=f'Начальная')
    ax1.axhline(y=ISOTHERM_TEMP, color='lime', linestyle='-', linewidth=2.5, alpha=0.8,
                label=f'Изотерма {ISOTHERM_TEMP} K')
    ax1.set_xlabel('Время (мкс)')
    ax1.set_ylabel('Температура (K)')
    ax1.set_title('Сравнение температуры в центре пучка')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    # Градиент температуры
    ax2 = fig.add_subplot(2, 2, 2)
    grad_pulsed = U_pulsed[center_x, center_y, surface_z, :] - U_pulsed[center_x, center_y, depth_idx, :]
    grad_cont = U_cont[center_x, center_y, surface_z, :] - U_cont[center_x, center_y, depth_idx, :]
    
    ax2.plot(t_phys, grad_pulsed, 'r-', linewidth=2.5, alpha=0.8, label='Импульсный')
    ax2.plot(t_phys, grad_cont, 'b-', linewidth=2.5, alpha=0.8, label='Непрерывный')
    ax2.set_xlabel('Время (мкс)')
    ax2.set_ylabel('Градиент температуры (K)')
    ax2.set_title('Градиент температуры (поверхность - глубина 30 мкм)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Объем нагретого материала
    ax3 = fig.add_subplot(2, 2, 3)
    vol_pulsed = [np.sum(U_pulsed[:, :, :, t] > ISOTHERM_TEMP) for t in range(len(t_plot))]
    vol_cont = [np.sum(U_cont[:, :, :, t] > ISOTHERM_TEMP) for t in range(len(t_plot))]
    
    max_vol = max(max(vol_pulsed), max(vol_cont))
    if max_vol > 0:
        vol_pulsed = np.array(vol_pulsed) / max_vol * 100
        vol_cont = np.array(vol_cont) / max_vol * 100
    
    ax3.plot(t_phys, vol_pulsed, 'r-', linewidth=2.5, alpha=0.8, label='Импульсный')
    ax3.plot(t_phys, vol_cont, 'b-', linewidth=2.5, alpha=0.8, label='Непрерывный')
    ax3.set_xlabel('Время (мкс)')
    ax3.set_ylabel('Относительный объем (%)')
    ax3.set_title(f'Объем материала с T > {ISOTHERM_TEMP}K')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.set_ylim(0, 105)
    
    # Интегральный перегрев
    ax4 = fig.add_subplot(2, 2, 4)
    int_pulsed = np.array([np.sum(temp_center_pulsed[:t+1] - ISOTHERM_TEMP) 
                          for t in range(len(t_plot)) if temp_center_pulsed[t] > ISOTHERM_TEMP])
    int_cont = np.array([np.sum(temp_center_cont[:t+1] - ISOTHERM_TEMP) 
                        for t in range(len(t_plot)) if temp_center_cont[t] > ISOTHERM_TEMP])
    
    ax4.plot(t_phys[:len(int_pulsed)], int_pulsed, 'r-', linewidth=2.5, alpha=0.8, label='Импульсный')
    ax4.plot(t_phys[:len(int_cont)], int_cont, 'b-', linewidth=2.5, alpha=0.8, label='Непрерывный')
    ax4.set_xlabel('Время (мкс)')
    ax4.set_ylabel('Интеграл перегрева (K·мкс)')
    ax4.set_title('Накопленный перегрев выше 1900K')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    plt.suptitle('Сравнительный анализ динамики лазерного нагрева: Импульсный vs Непрерывный', fontsize=15)
    plt.tight_layout()
    plt.savefig('animations/heating_dynamics_comparison.png', dpi=DPI, bbox_inches='tight')
    plt.show()


def plot_pulse_train_visualization():
    """Визуализация последовательности импульсов."""
    if config.LASER_MODE != "pulsed":
        print("Эта функция предназначена только для импульсного режима!")
        return
    
    total_time = config.NUM_PULSES * config.LASER_PULSE_PERIOD_NORM
    t_test = np.linspace(0, total_time, 5000)
    
    t_mod = t_test % config.LASER_PULSE_PERIOD_NORM
    source_values = config.LASER_AMPLITUDE * np.exp(
        -(t_mod - config.LASER_PULSE_PERIOD_NORM/2)**2 / (2 * config.LASER_PULSE_SIGMA_NORM**2))
    
    t_phys = t_test * config.CHARACTERISTIC_TIME * 1e6
    
    plt.figure(figsize=(14, 8))
    plt.plot(t_phys, source_values, 'r-', linewidth=2.5, alpha=0.8)
    plt.fill_between(t_phys, 0, source_values, alpha=0.2, color='red')
    
    for i in range(config.NUM_PULSES + 1):
        t_imp = i * config.LASER_PULSE_PERIOD_NORM * config.CHARACTERISTIC_TIME * 1e6
        plt.axvline(x=t_imp, color='gray', linestyle='--', alpha=0.5)
        if i < config.NUM_PULSES:
            plt.text(t_imp + 5, 0.9, f'{i+1}', fontsize=10, fontweight='bold', color='darkred')
            t_end = t_imp + config.LASER_PULSE_DURATION_NORM * config.CHARACTERISTIC_TIME * 1e6
            plt.axvspan(t_imp, t_end, alpha=0.1, color='red')
    
    level_1e2 = config.LASER_AMPLITUDE / np.e**2
    plt.axhline(y=level_1e2, color='orange', linestyle=':', alpha=0.7, linewidth=1.5,
                label=f'Уровень 1/e² ({level_1e2:.2f})')
    
    info = (f"ПАРАМЕТРЫ ПОСЛЕДОВАТЕЛЬНОСТИ ИМПУЛЬСОВ:\n" + "="*40 + "\n\n"
            f"• Количество импульсов: {config.NUM_PULSES}\n"
            f"• Период: {config.LASER_PULSE_PERIOD*1e6:.1f} мкс\n"
            f"• Длительность импульса: {config.LASER_PULSE_DURATION*1e6:.1f} мкс\n"
            f"• Частота: {config.LASER_REP_RATE:.0f} Гц\n"
            f"• Скважность: {config.LASER_DUTY_CYCLE*100:.1f}%\n"
            f"• Пиковая мощность: {config.LASER_PEAK_POWER:.1f} Вт\n"
            f"• Средняя мощность: {config.LASER_AVG_POWER} Вт\n"
            f"• Общее время: {total_time*config.CHARACTERISTIC_TIME*1e6:.1f} мкс")
    
    plt.text(0.02, 0.98, info, transform=plt.gca().transAxes, fontsize=10, va='top', linespacing=1.5,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9))
    
    plt.xlabel('Время (мкс)')
    plt.ylabel('Относительная интенсивность')
    plt.title(f'Последовательность лазерных импульсов\n{config.NUM_PULSES} импульсов, '
              f'период {config.LASER_PULSE_PERIOD*1e6:.1f} мкс, '
              f'длительность {config.LASER_PULSE_DURATION*1e6:.1f} мкс')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, config.LASER_AMPLITUDE * 1.05)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('animations/pulse_train_visualization.png', dpi=DPI, bbox_inches='tight')
    plt.show()


def plot_isotherm_1900_analysis(U_data, x_plot, y_plot, z_plot, t_plot, time_idx=-1):
    """Детальный анализ изотермы 1900 K."""
    U_physical = convert_to_physical_temperature(U_data)
    x_phys = np.array(x_plot) * config.CHARACTERISTIC_LENGTH * 1e6
    y_phys = np.array(y_plot) * config.CHARACTERISTIC_LENGTH * 1e6
    z_phys = np.array(z_plot) * config.CHARACTERISTIC_LENGTH * 1e6
    t_phys = t_plot[time_idx] * config.CHARACTERISTIC_TIME * 1e6
    
    center_x, center_y = len(x_plot)//2, len(y_plot)//2
    
    # Кратер (НОВАЯ ВЕРСИЯ)
    crater_field = laser_crater_3d(
        x_phys, y_phys, z_phys,
        max_depth_um=getattr(config, "CRATER_PEAK_DEPTH_UM", 30.0),
        crater_width_um=getattr(config, "CRATER_WIDTH_99_UM", 145.0),
        decay_length_um=getattr(config, "CRATER_DECAY_LENGTH_UM", 30.0)
    )
    
    surf_z_idx = np.argmin(np.abs(z_phys - 0.0))
    crater_surface = crater_field[:, :, surf_z_idx].T
    crater_peak = getattr(config, "CRATER_PEAK_DEPTH_UM", 30.0)
    level_99 = 0.01 * crater_peak
    
    fig = plt.figure(figsize=(18, 12))
    
    # Изотерма на поверхности
    ax1 = fig.add_subplot(2, 3, 1)
    X, Y = np.meshgrid(x_phys, y_phys)
    Z_surface = U_physical[:, :, -1, time_idx].T
    
    contour1 = ax1.contourf(X, Y, Z_surface, levels=20, cmap=TEMP_COLORMAP)
    
    if np.max(Z_surface) >= ISOTHERM_TEMP:
        cs = ax1.contour(X, Y, Z_surface, levels=[ISOTHERM_TEMP], colors='lime',
                        linewidths=3, alpha=0.9)
        for path in cs.get_paths():
            if len(path.vertices) > 2:
                polygon = Polygon(path.vertices, closed=True, fill=True,
                                 facecolor='lime', alpha=0.3, edgecolor='lime', linewidth=2)
                ax1.add_patch(polygon)
    
    ax1.contour(X, Y, crater_surface, levels=[level_99], colors='red',
                linewidths=2, linestyles='--', alpha=0.9)
    ax1.set_xlabel('x (мкм)')
    ax1.set_ylabel('y (мкм)')
    ax1.set_title(f'Изотерма {ISOTHERM_TEMP} K на поверхности\nВремя: {t_phys:.1f} мкс')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    fig.colorbar(contour1, ax=ax1, shrink=0.8, label='Температура (K)')
    
    # Профиль вдоль X
    ax2 = fig.add_subplot(2, 3, 2)
    x_profile = U_physical[:, center_y, -1, time_idx]
    ax2.plot(x_phys, x_profile, 'b-', linewidth=2.5)
    ax2.fill_between(x_phys, config.INITIAL_TEMPERATURE, x_profile, alpha=0.2, color='blue')
    ax2.axhline(y=ISOTHERM_TEMP, color='lime', linestyle='-', linewidth=2.5, alpha=0.8,
                label=f'Изотерма {ISOTHERM_TEMP} K')
    
    left = next((i for i in range(center_x, len(x_phys)) if x_profile[i] >= ISOTHERM_TEMP), None)
    right = next((i for i in range(center_x, len(x_phys)) if x_profile[i] < ISOTHERM_TEMP), None)
    if left and right:
        width = x_phys[right] - x_phys[left]
        ax2.fill_between(x_phys[left:right+1], ISOTHERM_TEMP, x_profile[left:right+1],
                        alpha=0.3, color='lime', label=f'Зона >{ISOTHERM_TEMP} K ({width:.1f} мкм)')
    
    ax2.set_xlabel('x (мкм)')
    ax2.set_ylabel('Температура (K)')
    ax2.set_title(f'Профиль температуры вдоль X (y=0)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Профиль по глубине
    ax3 = fig.add_subplot(2, 3, 3)
    depth_profile = U_physical[center_x, center_y, :, time_idx]
    ax3.plot(depth_profile, z_phys, 'r-', linewidth=2.5)
    ax3.axvline(x=ISOTHERM_TEMP, color='lime', linestyle='-', linewidth=2.5, alpha=0.8,
                label=f'Изотерма {ISOTHERM_TEMP} K')
    
    for i in range(len(z_phys)):
        if depth_profile[i] >= ISOTHERM_TEMP:
            depth = z_phys[i]
            ax3.fill_betweenx(z_phys[:i+1], ISOTHERM_TEMP, depth_profile[:i+1],
                            alpha=0.3, color='lime', label=f'Глубина: {depth:.1f} мкм')
            break
    
    ax3.set_xlabel('Температура (K)')
    ax3.set_ylabel('Глубина z (мкм)')
    ax3.set_title('Профиль температуры по глубине (центр)')
    ax3.grid(True, alpha=0.3)
    ax3.invert_yaxis()
    ax3.legend()
    
    # 3D визуализация
    ax4 = fig.add_subplot(2, 3, 4, projection='3d')
    mask = U_physical[:, :, :, time_idx] > ISOTHERM_TEMP
    
    if np.any(mask):
        x_idx, y_idx, z_idx = np.where(mask)
        sc = ax4.scatter(x_phys[x_idx], y_phys[y_idx], -z_phys[z_idx],
                        c=U_physical[x_idx, y_idx, z_idx, time_idx],
                        cmap=TEMP_COLORMAP, alpha=0.6, s=10, vmin=ISOTHERM_TEMP)
        ax4.text2D(0.05, 0.95, f'Точек >{ISOTHERM_TEMP}K: {len(x_idx)}',
                  transform=ax4.transAxes, fontsize=10, fontweight='bold',
                  bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        fig.colorbar(sc, ax=ax4, shrink=0.6, pad=0.1, label='Температура (K)')
    else:
        ax4.text2D(0.3, 0.5, f"Изотерма {ISOTHERM_TEMP} K\nне достигнута",
                  transform=ax4.transAxes, fontsize=12, fontweight='bold', color='red', ha='center')
    
    ax4.set_xlabel('x (мкм)')
    ax4.set_ylabel('y (мкм)')
    ax4.set_zlabel('Глубина (мкм)')
    ax4.set_title(f'3D распределение зоны >{ISOTHERM_TEMP} K')
    ax4.view_init(elev=25, azim=45)
    
    # Эволюция размеров
    ax5 = fig.add_subplot(2, 3, 5)
    widths, depths, volumes = [], [], []
    
    for t_idx in range(len(t_plot)):
        surface_temp = U_physical[:, center_y, -1, t_idx]
        left = next((i for i in range(center_x, len(x_phys)) if surface_temp[i] >= ISOTHERM_TEMP), None)
        right = next((i for i in range(center_x, len(x_phys)) if surface_temp[i] < ISOTHERM_TEMP), None)
        widths.append(x_phys[right] - x_phys[left] if left and right else 0)
        
        center_temp = U_physical[center_x, center_y, :, t_idx]
        depths.append(next((z_phys[i] for i in range(len(z_phys)) if center_temp[i] >= ISOTHERM_TEMP), 0))
        volumes.append(np.sum(U_physical[:, :, :, t_idx] > ISOTHERM_TEMP))
    
    volumes = np.array(volumes) / max(volumes) * 100 if max(volumes) > 0 else volumes
    
    ax5.plot(t_phys, widths, 'b-', linewidth=2, label='Ширина зоны (мкм)')
    ax5.plot(t_phys, depths, 'r-', linewidth=2, label='Глубина (мкм)')
    ax5.plot(t_phys, volumes, 'g-', linewidth=2, label='Отн. объем (%)')
    ax5.set_xlabel('Время (мкс)')
    ax5.set_ylabel('Параметры зоны')
    ax5.set_title(f'Эволюция зоны >{ISOTHERM_TEMP} K')
    ax5.grid(True, alpha=0.3)
    ax5.legend()
    
    # Информация
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    
    current_width = widths[time_idx] if time_idx < len(widths) else widths[-1]
    current_depth = depths[time_idx] if time_idx < len(depths) else depths[-1]
    current_vol = volumes[time_idx] if time_idx < len(volumes) else volumes[-1]
    
    info = (f"АНАЛИЗ ИЗОТЕРМЫ {ISOTHERM_TEMP} K\n" + "="*30 + f"\n\nТекущее время: {t_phys:.1f} мкс\n\n"
            f"ТЕКУЩИЕ ПАРАМЕТРЫ:\n"
            f"• Макс. темп.: {np.max(U_physical[:, :, :, time_idx]):.1f} K\n"
            f"• Ширина зоны: {current_width:.1f} мкм\n"
            f"• Глубина: {current_depth:.1f} мкм\n"
            f"• Отн. объем: {current_vol:.1f}%\n\n"
            f"МАКСИМАЛЬНЫЕ ЗНАЧЕНИЯ:\n"
            f"• Макс. ширина: {np.max(widths):.1f} мкм\n"
            f"• Макс. глубина: {np.max(depths):.1f} мкм\n"
            f"• Время макс. объема: {t_phys[np.argmax(volumes)]:.1f} мкс")
    
    ax6.text(0.1, 0.95, info, fontsize=11, va='top', linespacing=1.5,
             bbox=dict(boxstyle="round,pad=0.6", facecolor="lightgreen", alpha=0.9))
    
    mode_text = "Импульсный" if config.LASER_MODE == "pulsed" else "Непрерывный"
    plt.suptitle(f'Детальный анализ изотермы {ISOTHERM_TEMP} K\nРежим: {mode_text}, Время: {t_phys:.1f} мкс',
                fontsize=15, y=0.98)
    
    plt.tight_layout()
    plt.savefig(f'animations/isotherm_{ISOTHERM_TEMP}_analysis_{mode_text}_t{t_phys:.0f}us.png',
               dpi=DPI, bbox_inches='tight')
    plt.show()