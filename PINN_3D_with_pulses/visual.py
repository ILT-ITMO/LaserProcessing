import numpy as np
import matplotlib.pyplot as plt
import config
from matplotlib.animation import FuncAnimation, PillowWriter



def visualize_laser_pulses():
    """Визуализация временного профиля лазерных импульсов"""
    t_test = np.linspace(0, 0.5, 1000)
    source_values = np.zeros_like(t_test)
    
    for i, t_val in enumerate(t_test):
        t_mod = t_val % config.LASER_PULSE_PERIOD
        if 0 <= t_mod <= config.LASER_PULSE_DURATION:
            source_values[i] = config.LASER_AMPLITUDE
    
    plt.figure(figsize=(10, 4))
    plt.plot(t_test, source_values)
    plt.xlabel('Время (нормализованное)')
    plt.ylabel('Амплитуда лазера')
    plt.title('Временной профиль лазерных импульсов')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('animations/laser_pulse_profile.png')
    plt.show()


def visualize_laser_spatial_profile():
    """Визуализация пространственного профиля лазерного пучка"""
    x_test = np.linspace(-1, 1, 100)
    y_test = np.linspace(-1, 1, 100)
    X, Y = np.meshgrid(x_test, y_test)
    
    spatial_dist = config.LASER_AMPLITUDE * np.exp(-(X**2 + Y**2) / (2 * config.LASER_SIGMA**2))
    
    plt.figure(figsize=(8, 6))
    plt.contourf(X, Y, spatial_dist, levels=50, cmap='hot')
    plt.colorbar(label='Интенсивность')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Пространственное распределение лазерного пучка')
    plt.tight_layout()
    plt.savefig('animations/laser_spatial_profile.png')
    plt.show()


def create_animation(U_data, x_plot, y_plot, z_plot, t_plot, title, filename):
    """
    Создает анимацию с информацией о лазерных импульсах
    """
    fig = plt.figure(figsize=(20, 8))
    
    def update(frame):
        fig.clear()
        
        for i, (slice_idx, plane) in enumerate([(len(z_plot)//2, 'XY'), 
                                            (len(y_plot)//2, 'XZ'), 
                                            (len(x_plot)//2, 'YZ')]):
            ax = fig.add_subplot(2, 3, i+1)
            
            if plane == 'XY':
                data = U_data[:, :, slice_idx, frame].T
                extent = [x_plot[0], x_plot[-1], y_plot[0], y_plot[-1]]
                xlabel, ylabel = 'x', 'y'
            elif plane == 'XZ':
                data = U_data[:, slice_idx, :, frame].T
                extent = [x_plot[0], x_plot[-1], z_plot[0], z_plot[-1]]
                xlabel, ylabel = 'x', 'z'
            else:  # YZ
                data = U_data[slice_idx, :, :, frame].T
                extent = [y_plot[0], y_plot[-1], z_plot[0], z_plot[-1]]
                xlabel, ylabel = 'y', 'z'
            
            im = ax.imshow(data, extent=extent, origin='lower', 
                        aspect='auto', cmap='jet', vmin=0, vmax=np.max(U_data))
            ax.set_title(f'{plane} срез')
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            plt.colorbar(im, ax=ax, shrink=0.8)
        
        # График временного профиля лазера
        ax_laser = fig.add_subplot(2, 3, 4)
        t_range = np.linspace(0, max(t_plot), 1000)
        laser_profile = np.zeros_like(t_range)
        for i, t_val in enumerate(t_range):
            t_mod = t_val % config.LASER_PULSE_PERIOD
            if 0 <= t_mod <= config.LASER_PULSE_DURATION:
                laser_profile[i] = config.LASER_AMPLITUDE
        
        ax_laser.plot(t_range, laser_profile, 'r-', linewidth=2)
        ax_laser.axvline(x=t_plot[frame], color='blue', linestyle='--', alpha=0.7)
        ax_laser.set_xlabel('Время')
        ax_laser.set_ylabel('Интенсивность лазера')
        ax_laser.set_title('Лазерные импульсы')
        ax_laser.grid(True)
        
        # Информация о текущем импульсе
        ax_info = fig.add_subplot(2, 3, 5)
        ax_info.axis('off')
        current_time = t_plot[frame]
        pulse_number = int(current_time // config.LASER_PULSE_PERIOD) + 1
        time_in_pulse = current_time % config.LASER_PULSE_PERIOD
        
        info_text = f"Время: {current_time:.3f}\n"
        info_text += f"Импульс №: {pulse_number}\n"
        info_text += f"Время в импульсе: {time_in_pulse:.3f}\n"
        if time_in_pulse <= config.LASER_PULSE_DURATION:
            info_text += "Лазер: ВКЛ"
        else:
            info_text += "Лазер: ВЫКЛ"
        
        ax_info.text(0.1, 0.5, info_text, fontsize=12, va='center')
        
        # Пространственный профиль лазера
        ax_profile = fig.add_subplot(2, 3, 6)
        x_profile = np.linspace(-1, 1, 100)
        laser_spatial = config.LASER_AMPLITUDE * np.exp(-x_profile**2 / (2 * config.LASER_SIGMA**2))
        ax_profile.plot(x_profile, laser_spatial, 'g-', linewidth=2)
        ax_profile.set_xlabel('x')
        ax_profile.set_ylabel('Интенсивность')
        ax_profile.set_title('Профиль лазерного пучка')
        ax_profile.grid(True)
        
        plt.suptitle(f'{title}\nTime: {t_plot[frame]:.3f}', fontsize=16)
        plt.tight_layout()
        return fig,

    ani = FuncAnimation(fig, update, frames=len(t_plot), interval=500, blit=False, repeat=True)
    
    writer = PillowWriter(fps=2)
    ani.save(filename, writer=writer, dpi=100)
    plt.close(fig)
    return ani