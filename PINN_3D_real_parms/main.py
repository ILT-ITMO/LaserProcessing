import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import json
from visual import visualize_laser_pulses, visualize_laser_spatial_profile, create_animation, laser_crater_3d
from pinn import PINN, train_pinn
from conditions import convert_to_physical_coords, convert_to_physical_temperature
import config

def run_simulation(config_file=None, laser_mode=None):
    """
    Запуск симуляции для моделирования теплового воздействия лазера на материал.
    
    Метод загружает конфигурацию, определяет устройство для вычислений (CPU, CUDA, MPS),
    визуализирует профили лазерных импульсов, обучает модель PINN для решения уравнения теплопроводности
    и выполняет предсказания для получения распределения температуры. Результаты сохраняются в виде анимации и графиков.
    
    Args:
        config_file (str, optional): Путь к JSON файлу конфигурации. Defaults to None.
        laser_mode (str, optional): Явное указание режима лазера ("pulsed" или "continuous"). Defaults to None.
    
    Returns:
        tuple: Кортеж, содержащий обученную модель PINN, предсказанное поле температуры и историю потерь.
    """
    # Перезагружаем конфигурацию если указан файл
    if config_file:
        print(f"Загрузка конфигурации из {config_file}")
        config.CONFIG.load_from_json(config_file)
        config.CONFIG.calculate_derived_parameters()
        config.CONFIG.print_summary()
    
    # Если явно указан режим, обновляем его
    if laser_mode is not None:
        config.CONFIG.config["laser"]["mode"] = laser_mode
        config.CONFIG.calculate_derived_parameters()
        print(f"Режим явно установлен на: {laser_mode}")
    
    # Определяем устройство
    try:
        if config.CONFIG.config["training"]["device"] == "auto":
            device = torch.device("mps" if torch.backends.mps.is_available() else 
                                 "cuda" if torch.backends.cuda.is_built() else "cpu")
        else:
            device = torch.device(config.CONFIG.config["training"]["device"])
    except:
        device = torch.device("cuda" if torch.backends.cuda.is_built() else "cpu")
    
    print(f"Используемое устройство: {device}")
    
    # Получаем текущий режим
    mode = config.LASER_MODE
    print(f"Текущий режим лазера: {mode}")

    # Создаем папку для результатов
    output_dir = f'results/{mode}'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs('animations', exist_ok=True)

    # Визуализация профилей
    print("\nВизуализация профиля лазерных импульсов...")
    visualize_laser_pulses()
    visualize_laser_spatial_profile()

    # Создаем и обучаем модель
    model = PINN([4, 128, 128, 128, 1]).to(device)
    diff_coef = float(getattr(config, "INITIAL_DIFF_COEF", 1.0))

    print(f"\nОбучение 3D PINN в режиме {mode}...")
    loss_hist = train_pinn(
        model, 
        diff_coef, 
        num_epochs=config.CONFIG.config["training"]["num_epochs"],
        lr=config.CONFIG.config["training"]["learning_rate"],
        device=device,
        laser_mode=mode,
        pretrained_checkpoint_path=config.CONFIG.config.get("training", {}).get("pretrained_checkpoint_path"),
        pretrained_strict=config.CONFIG.config.get("training", {}).get("pretrained_strict", True),
    )

    # Подготовка данных для предсказания
    viz_points = config.CONFIG.config["pinn"]["visualization_points"]
    nx_plot, ny_plot, nz_plot, nt_plot = (
        viz_points["x"], viz_points["y"], viz_points["z"], viz_points["t"]
    )
    
    x_plot = np.linspace(-1, 1, nx_plot)  
    y_plot = np.linspace(-1, 1, ny_plot)  
    z_plot = np.linspace(0, 1, nz_plot)   
    t_plot = np.linspace(0, config.SIMULATION_TIME_NORM, nt_plot)  
    
    print("Выполнение предсказаний PINN...")
    with torch.no_grad():
        Xp, Yp, Zp, Tp = np.meshgrid(x_plot, y_plot, z_plot, t_plot, indexing='ij')
        x_t = torch.tensor(Xp.flatten(), dtype=torch.float32, device=device)
        y_t = torch.tensor(Yp.flatten(), dtype=torch.float32, device=device)
        z_t = torch.tensor(Zp.flatten(), dtype=torch.float32, device=device)
        t_t = torch.tensor(Tp.flatten(), dtype=torch.float32, device=device)
        
        U_pred_norm = model(x_t, y_t, z_t, t_t).cpu().numpy().reshape(
            nx_plot, ny_plot, nz_plot, nt_plot
        )
        
        # Конвертация в физические величины
        U_pred_physical = convert_to_physical_temperature(U_pred_norm)

    metrics = None
    # ---------------------------------------------------------------------
    # Метрики (MSE/MAE) между смоделированной изотермой 1900K и линией кратера
    # ---------------------------------------------------------------------
    try:
        # Физические координаты (мкм) для сетки визуализации
        x_phys = np.array(x_plot) * config.CHARACTERISTIC_LENGTH * 1e6
        y_phys = np.array(y_plot) * config.CHARACTERISTIC_LENGTH * 1e6
        z_phys = np.array(z_plot) * config.CHARACTERISTIC_LENGTH * 1e6

        time_idx = -1  # после окончания моделирования
        isotherm_temp = float(getattr(config, "CRATER_TARGET_TEMP", 1900.0))

        center_y = len(y_phys) // 2
        surface_z_idx = len(z_phys) - 1  # в этой сетке поверхность соответствует z=0 -> последний индекс

        simulated_line = U_pred_physical[:, center_y, surface_z_idx, time_idx].astype(np.float64)

        crater_field = laser_crater_3d(
            x_phys, y_phys, z_phys,
            max_depth_um=getattr(config, "CRATER_PEAK_DEPTH_UM", 30.0),
            crater_width_um=getattr(config, "CRATER_WIDTH_99_UM", 145.0),
            max_height_um=getattr(config, "CRATER_PEAK_DEPTH_UM", 30.0),
            decay_length_um=getattr(config, "CRATER_DECAY_LENGTH_UM", 30.0),
        )
        crater_surface = crater_field[:, :, np.argmin(np.abs(z_phys - 0.0))]
        crater_peak = float(getattr(config, "CRATER_PEAK_DEPTH_UM", 30.0))
        crater_line = crater_surface[:, center_y].astype(np.float64)
        experimental_line = (isotherm_temp - crater_peak + crater_line).astype(np.float64)

        # На всякий случай приводим к общему размеру
        n = int(min(simulated_line.shape[0], experimental_line.shape[0]))
        simulated_line = simulated_line[:n]
        experimental_line = experimental_line[:n]
        x_phys_line = x_phys[:n]

        mse = float(np.mean((simulated_line - experimental_line) ** 2))
        mae = float(np.mean(np.abs(simulated_line - experimental_line)))

        metrics = {
            "mode": str(mode),
            "time_idx": int(time_idx),
            "isotherm_temp_K": float(isotherm_temp),
            "mse": mse,
            "mae": mae,
        }

        metrics_path = os.path.join(output_dir, f"metrics_{mode}.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

        print("\nМетрики сравнения (смоделированная линия vs кратерная линия):")
        print(f"  MSE = {mse:.6g}")
        print(f"  MAE = {mae:.6g}")
        print(f"  Файл: {metrics_path}")

        # Дополнительно сохраняем сравниваемые линии для последующего анализа
        np.save(os.path.join(output_dir, f"metrics_x_um_{mode}.npy"), x_phys_line)
        np.save(os.path.join(output_dir, f"metrics_simulated_line_K_{mode}.npy"), simulated_line)
        np.save(os.path.join(output_dir, f"metrics_experimental_line_K_{mode}.npy"), experimental_line)
    except Exception as e:
        print(f"\n[WARN] Не удалось посчитать MSE/MAE: {e}")

    print("\nСоздание анимаций...")
    mode_title = "Непрерывный" if mode == "continuous" else "Импульсный"
    title = f'PINN Solution: Нагрев кварца лазером\n{mode_title} режим, СТАТИЧНЫЙ пучок'
    filename = f'animations/pinn_solution_{mode}.gif'
    
    create_animation(U_pred_norm, x_plot, y_plot, z_plot, t_plot, title, filename, metrics=metrics)

    # График обучения
    plt.figure(figsize=(8,5))
    plt.plot(loss_hist)
    plt.yscale('log')
    plt.title(f'Кривая обучения ({mode.capitalize()} режим)')
    plt.xlabel('Эпоха')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/learning_curve_{mode}.png')
    plt.show()

    print(f"\nВсе результаты сохранены в папке '{output_dir}/'")
    print(f"Максимальная температура: {np.max(U_pred_physical):.1f} K")
    print(f"Перегрев: {np.max(U_pred_physical) - config.INITIAL_TEMPERATURE:.1f} K")
    print(f"Физическое время моделирования: {config.SIMULATION_TIME_PHYSICAL*1e6:.1f} мкс")
    
    # Сохраняем результаты
    np.save(f'{output_dir}/temperature_field_{mode}.npy', U_pred_physical)
    np.save(f'{output_dir}/loss_history_{mode}.npy', loss_hist)
    
    return model, U_pred_physical, loss_hist

if __name__ == '__main__':
    # Проверяем аргументы командной строки
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
        if os.path.exists(config_file):
            # Можно указать режим как второй аргумент
            laser_mode = sys.argv[2] if len(sys.argv) > 2 else None
            run_simulation(config_file, laser_mode)
        else:
            print(f"Файл конфигурации {config_file} не найден")
            run_simulation()  # запуск с конфигурацией по умолчанию
    else:
        # Запуск с конфигурацией по умолчанию
        print("Запуск с конфигурацией по умолчанию")
        print("Для использования JSON конфигурации: python main.py config.json")
        print("Для указания режима: python main.py config.json pulsed|continuous")
        run_simulation()