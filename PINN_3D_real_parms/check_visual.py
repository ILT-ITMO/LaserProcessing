import numpy as np
import matplotlib.pyplot as plt

t_test = np.linspace(0, 0,6, 1000)
source_values = np.ones_like(t_test) * 1

# Конвертация в физическое время
t_phys = t_test * 0.000004 * 1e6  # в микросекунды

plt.figure(figsize=(10, 4))
plt.plot(t_phys, source_values, 'b-', linewidth=2)
plt.fill_between(t_phys, 0, source_values, alpha=0.3, color='blue')
plt.xlabel('Время (мкс)')
plt.ylabel('Относительная интенсивность')
plt.title('Временной профиль лазерного излучения (Непрерывный режим)')
plt.grid(True, alpha=0.3)
plt.ylim(0, 1 * 1.1)

# Добавим информацию о мощности
plt.text(0.02, 0.95, f'Мощность: {5} Вт', 
        transform=plt.gca().transAxes, fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))

plt.tight_layout()
# plt.savefig('animations/laser_continuous_profile.png', dpi=150)
plt.show()