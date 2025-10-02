# visual.py
import numpy as np
import matplotlib.pyplot as plt
import config
from matplotlib.animation import FuncAnimation, PillowWriter
import physical_params as phys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import physical_params as phys
import config
from conditions import laser_source_term  # Импортируем правильную функцию источника

def visualize_laser_pulses():
    """Визуализация временного профиля лазерных импульсов в размерных величинах"""
    t_test = np.linspace(0, phys.T_MAX, 1000)  # секунды
    
    # Используем правильный временной профиль (гауссов вместо прямоугольного)
    source_values = np.zeros_like(t_test)
    
    for i, t_val in enumerate(t_test):
        t_mod = t_val % phys.LASER_PULSE_PERIOD
        # Гауссов импульс вместо прямоугольного
        pulse_center = phys.LASER_PULSE_DURATION / 2
        temporal_dist = np.exp(-(t_mod - pulse_center)**2 / (2 * (phys.LASER_PULSE_DURATION/4)**2))
        source_values[i] = phys.LASER_AMPLITUDE * temporal_dist
    
    plt.figure(figsize=(12, 4))
    plt.plot(t_test * 1e6, source_values / 1e9)  # мкс и ГВт/м³
    plt.xlabel('Время (мкс)')
    plt.ylabel('Мощность источника (ГВт/м³)')
    plt.title('Временной профиль лазерных импульсов (гауссов) - кварц JGS1')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('animations/laser_pulse_profile_quartz.png', dpi=150)
    plt.show()

def visualize_laser_spatial_profile():
    """Визуализация пространственного профиля лазерного пучка в размерных величинах"""
    x_test = np.linspace(phys.X_MIN, phys.X_MAX, 100) * 1e6  # мкм
    y_test = np.linspace(phys.Y_MIN, phys.Y_MAX, 100) * 1e6  # мкм
    X, Y = np.meshgrid(x_test, y_test)
    
    # Преобразуем обратно в метры для расчета
    X_m = X * 1e-6
    Y_m = Y * 1e-6
    
    # Центрируем координаты (пучок в центре области)
    x_center = (phys.X_MAX + phys.X_MIN) / 2
    y_center = (phys.Y_MAX + phys.Y_MIN) / 2
    X_centered = X_m - x_center
    Y_centered = Y_m - y_center
    
    # Пространственное распределение (гауссов пучок)
    spatial_dist = phys.LASER_AMPLITUDE * np.exp(-(X_centered**2 + Y_centered**2) / (2 * phys.LASER_SIGMA**2))
    
    plt.figure(figsize=(10, 8))
    contour = plt.contourf(X, Y, spatial_dist / 1e9, levels=50, cmap='hot')  # ГВт/м³
    plt.colorbar(contour, label='Интенсивность (ГВт/м³)')
    plt.xlabel('x (мкм)')
    plt.ylabel('y (мкм)')
    plt.title('Пространственное распределение лазерного пучка - кварц JGS1')
    plt.tight_layout()
    plt.savefig('animations/laser_spatial_profile_quartz.png', dpi=150)
    plt.show()

def visualize_absorption_depth_profile():
    """Визуализация экспоненциального поглощения по глубине согласно закону Бугера-Ламберта"""
    z_depth = np.linspace(phys.Z_MIN, phys.Z_MAX, 100) * 1e6  # мкм
    z_norm = np.linspace(0, 1, 100)  # безразмерная глубина [0,1]
    
    # Экспоненциальное поглощение: q(z) = μ_* * exp(-μ_* * z)
    absorption_profile = phys.MU_STAR * np.exp(-phys.MU_STAR * z_norm)
    
    plt.figure(figsize=(10, 6))
    plt.plot(z_depth, absorption_profile, 'b-', linewidth=2)
    plt.fill_between(z_depth, 0, absorption_profile, alpha=0.3, color='blue')
    plt.xlabel('Глубина z (мкм)')
    plt.ylabel('Относительная мощность поглощения')
    plt.title(f'Экспоненциальное поглощение по глубине (закон Бугера-Ламберта)\nμ* = {phys.MU_STAR:.2f}')
    plt.grid(True, alpha=0.3)
    
    # Добавляем информацию о глубине проникновения
    penetration_depth = 1 / phys.MU_STAR  # безразмерная глубина проникновения
    penetration_depth_physical = penetration_depth * (phys.Z_MAX - phys.Z_MIN) * 1e6  # мкм
    
    plt.axvline(x=penetration_depth_physical, color='red', linestyle='--', 
                label=f'Глубина проникновения: {penetration_depth_physical:.1f} мкм')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('animations/absorption_depth_profile.png', dpi=150)
    plt.show()

def create_animation(results, normalizer, title, filename):
    """
    Создает анимацию температурного распределения в РАЗМЕРНЫХ величинах
    results - словарь с результатами в размерных величинах из postprocess_results()
    """
    # Извлекаем размерные данные
    x_phys = results['x_physical'] * 1e6  # мкм
    y_phys = results['y_physical'] * 1e6  # мкм
    z_phys = results['z_physical'] * 1e6  # мкм
    t_phys = results['t_physical'] * 1e6  # мкс
    T_phys = results['temperature']       # K
    
    # Находим индексы центров
    x_center_idx = len(x_phys) // 2
    y_center_idx = len(y_phys) // 2
    z_surface_idx = 0  # поверхность
    z_mid_idx = len(z_phys) // 2  # середина толщины
    
    fig = plt.figure(figsize=(22, 12))
    
    def update(frame):
        fig.clear()
        
        # 1. XY срез на поверхности (z = 0)
        ax1 = fig.add_subplot(2, 4, 1)
        data_xy = T_phys[:, :, z_surface_idx, frame].T
        X_xy, Y_xy = np.meshgrid(x_phys, y_phys)
        
        contour1 = ax1.contourf(X_xy, Y_xy, data_xy, levels=50, cmap='hot')
        ax1.set_title('XY срез (поверхность)')
        ax1.set_xlabel('x, мкм')
        ax1.set_ylabel('y, мкм')
        ax1.set_aspect('equal')
        plt.colorbar(contour1, ax=ax1, shrink=0.8, label='Температура, K')
        
        # Отметка центра пучка
        ax1.axhline(y=0, color='white', linestyle='--', alpha=0.5, linewidth=0.5)
        ax1.axvline(x=0, color='white', linestyle='--', alpha=0.5, linewidth=0.5)
        ax1.plot(0, 0, 'w+', markersize=10, markeredgewidth=2)
        
        # 2. XZ срез через центр (y = 0)
        ax2 = fig.add_subplot(2, 4, 2)
        data_xz = T_phys[:, y_center_idx, :, frame].T
        X_xz, Z_xz = np.meshgrid(x_phys, z_phys)
        
        contour2 = ax2.contourf(X_xz, Z_xz, data_xz, levels=50, cmap='hot')
        ax2.set_title('XZ срез (через центр пучка)')
        ax2.set_xlabel('x, мкм')
        ax2.set_ylabel('z, мкм')
        plt.colorbar(contour2, ax=ax2, shrink=0.8, label='Температура, K')
        ax2.invert_yaxis()  # глубина увеличивается вниз
        
        # 3. YZ срез через центр (x = 0)
        ax3 = fig.add_subplot(2, 4, 3)
        data_yz = T_phys[x_center_idx, :, :, frame].T
        Y_yz, Z_yz = np.meshgrid(y_phys, z_phys)
        
        contour3 = ax3.contourf(Y_yz, Z_yz, data_yz, levels=50, cmap='hot')
        ax3.set_title('YZ срез (через центр пучка)')
        ax3.set_xlabel('y, мкм')
        ax3.set_ylabel('z, мкм')
        plt.colorbar(contour3, ax=ax3, shrink=0.8, label='Температура, K')
        ax3.invert_yaxis()  # глубина увеличивается вниз
        
        # 4. Профиль температуры вдоль x на поверхности
        ax4 = fig.add_subplot(2, 4, 4)
        T_profile_x = T_phys[:, y_center_idx, z_surface_idx, frame]
        
        ax4.plot(x_phys, T_profile_x, 'b-', linewidth=2)
        ax4.set_xlabel('x, мкм')
        ax4.set_ylabel('Температура, K')
        ax4.set_title('Профиль по x на поверхности')
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, np.max(T_phys[:, y_center_idx, z_surface_idx, :]) * 1.1)
        
        # Отметка положения лазерного пучка
        laser_radius = phys.LASER_SIGMA * 1e6  # мкм
        ax4.axvspan(-laser_radius, laser_radius, alpha=0.2, color='red', label='Область пучка')
        ax4.legend()
        
        # 5. Распределение температуры по глубине в центре
        ax5 = fig.add_subplot(2, 4, 5)
        T_depth = T_phys[x_center_idx, y_center_idx, :, frame]
        
        ax5.plot(T_depth, z_phys, 'g-', linewidth=2)
        ax5.set_xlabel('Температура, K')
        ax5.set_ylabel('Глубина z, мкм')
        ax5.set_title('Распределение по глубине в центре')
        ax5.grid(True, alpha=0.3)
        ax5.invert_yaxis()  # глубина увеличивается вниз
        
        # Теоретическая кривая экспоненциального затухания
        max_surface_temp = T_phys[x_center_idx, y_center_idx, z_surface_idx, frame]
        theoretical_depth = max_surface_temp * np.exp(-phys.MU_STAR * (z_phys / (z_phys[-1] - z_phys[0])))
        ax5.plot(theoretical_depth, z_phys, 'r--', alpha=0.7, linewidth=1.5, label='Теор. затухание')
        ax5.legend()
        
        # 6. Временная эволюция температуры в центре
        ax6 = fig.add_subplot(2, 4, 6)
        T_center_evolution = T_phys[x_center_idx, y_center_idx, z_surface_idx, :]
        
        ax6.plot(t_phys[:frame+1], T_center_evolution[:frame+1], 'r-', linewidth=2)
        ax6.axvline(x=t_phys[frame], color='blue', linestyle='--', alpha=0.7, linewidth=2)
        ax6.set_xlabel('Время, мкс')
        ax6.set_ylabel('Температура, K')
        ax6.set_title('Температура в центре пятна')
        ax6.grid(True, alpha=0.3)
        ax6.set_xlim(0, t_phys[-1])
        ax6.set_ylim(0, np.max(T_center_evolution) * 1.1)
        
        # Отметка лазерных импульсов
        for i in range(0, int(t_phys[-1]) + 1, int(phys.LASER_PULSE_PERIOD * 1e6)):
            ax6.axvspan(i, i + phys.LASER_PULSE_DURATION * 1e6, 
                       alpha=0.2, color='red', label='Лазер' if i == 0 else "")
        
        if frame == 0:
            ax6.legend()
        
        # 7. Информационная панель
        ax7 = fig.add_subplot(2, 4, 7)
        ax7.axis('off')
        
        current_time_us = t_phys[frame]
        pulse_number = int(current_time_us // (phys.LASER_PULSE_PERIOD * 1e6)) + 1
        time_in_pulse_us = current_time_us % (phys.LASER_PULSE_PERIOD * 1e6)
        
        # Определяем, активен ли лазер (гауссов профиль)
        pulse_center = phys.LASER_PULSE_DURATION * 1e6 / 2
        pulse_active = (abs(time_in_pulse_us - pulse_center) < phys.LASER_PULSE_DURATION * 1e6 / 2)
        
        max_temp = np.max(T_phys[..., frame])
        min_temp = np.min(T_phys[..., frame])
        center_temp = T_phys[x_center_idx, y_center_idx, z_surface_idx, frame]
        
        info_text = f"Время: {current_time_us:.1f} мкс\n"
        info_text += f"Импульс №: {pulse_number}\n"
        info_text += f"В импульсе: {time_in_pulse_us:.1f} мкс\n\n"
        info_text += f"Температура в центре: {center_temp:.1f} K\n"
        info_text += f"Макс. температура: {max_temp:.1f} K\n"
        info_text += f"Мин. температура: {min_temp:.1f} K\n\n"
        info_text += f"Параметры:\n"
        info_text += f"μ* = {phys.MU_STAR:.2f}\n"
        info_text += f"w₀ = {phys.LASER_SIGMA*1e6:.1f} мкм\n"
        info_text += f"τ_imp = {phys.LASER_PULSE_DURATION*1e6:.1f} мкс\n"
        
        if pulse_active:
            info_text += "\n🔴 ЛАЗЕР: АКТИВЕН"
            ax7.set_facecolor('#FFF0F0')
        else:
            info_text += "\n⚫ ЛАЗЕР: ВЫКЛ"
            ax7.set_facecolor('#F0F0F0')
        
        ax7.text(0.05, 0.95, info_text, fontsize=10, va='top', 
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9),
                transform=ax7.transAxes)
        
        # 8. Пространственный профиль лазерного пучка
        ax8 = fig.add_subplot(2, 4, 8)
        x_profile = np.linspace(-50, 50, 100)  # мкм
        # Гауссов профиль интенсивности
        laser_intensity = np.exp(-2 * (x_profile * 1e-6)**2 / (phys.LASER_SIGMA**2))
        ax8.plot(x_profile, laser_intensity * 100, 'r-', linewidth=2)
        ax8.fill_between(x_profile, 0, laser_intensity * 100, alpha=0.3, color='red')
        ax8.set_xlabel('Радиус, мкм')
        ax8.set_ylabel('Интенсивность, %')
        ax8.set_title('Профиль лазерного пучка')
        ax8.grid(True, alpha=0.3)
        ax8.set_ylim(0, 110)
        
        # Отметка радиуса пучка
        ax8.axvline(x=phys.LASER_SIGMA*1e6, color='red', linestyle='--', alpha=0.5, label='w₀')
        ax8.axvline(x=-phys.LASER_SIGMA*1e6, color='red', linestyle='--', alpha=0.5)
        ax8.legend()
        
        plt.suptitle(f'{title}\n'
                    f'Время: {current_time_us:.1f} мкс ∙ Импульс: {pulse_number} ∙ '
                    f'Температура в центре: {center_temp:.1f} K ∙ μ* = {phys.MU_STAR:.2f}', 
                    fontsize=14, y=0.98)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.94)
        
        return fig,

    # Создаем анимацию (уменьшаем количество кадров для скорости)
    frames_step = max(1, len(t_phys) // 50)  # не более 50 кадров
    frames_indices = range(0, len(t_phys), frames_step)
    
    def update_wrapper(frame_idx):
        return update(frame_idx)
    
    ani = FuncAnimation(fig, update_wrapper, frames=frames_indices, interval=200, blit=False, repeat=True)
    
    # Сохраняем с высоким качеством
    writer = PillowWriter(fps=5, bitrate=2000)
    ani.save(filename, writer=writer, dpi=120)
    plt.close(fig)
    
    print(f"Анимация в размерных величинах сохранена: {filename}")
    return ani

def create_comparison_animation(results_before, results_after, normalizer, filename):
    """
    Создает анимацию сравнения до и после исправления архитектуры
    """
    # Реализация функции сравнения (можно добавить позже)
    pass

# Добавляем вызов новой функции визуализации в существующие функции
def visualize_all_laser_profiles():
    """Визуализация всех профилей лазера"""
    print("Визуализация профилей лазерного излучения...")
    visualize_laser_pulses()
    visualize_laser_spatial_profile()
    visualize_absorption_depth_profile()  # Новая функция
    print("Все профили сохранены в папке 'animations/'")

