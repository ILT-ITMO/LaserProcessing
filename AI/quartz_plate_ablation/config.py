# config.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class Config:
    # --- Геометрия и нормировки (м) ---
    R_m: float = 300e-6           # радиус области по r
    H_m: float = 100e-6           # высота области по z
    w0_m: float = 62e-6          # радиус пучка по e^-2 (на поверхности)
    Twindow_s: float = 1e-6       # временное окно моделирования
    mu_star: float = 10.0         # безразмерный поглотительный параметр (μ_a * H)
    wl: float = 10.6e-6            # длина волны излучения
    # --- Материал ---
    rho_kg_m3: float = 2200.0     # плотность (кварц)
    cp_J_kgK: float = 740.0       # теплоёмкость
    k_W_mK: float = 1.4           # теплопроводность
    T0_C: float = 20.0            # начальная температура (°C)
    k_imag: float = 0.024         # мнимая часть коэфициента поглощения стекла для 10.6 мкм излучения

    # --- Нормировка температуры ---
    # Вычисляется в create_preset_param.py и кладётся в JSON как DELTA_T_SCALE_K
    temp_scaling_mode: str = "A"  # "A" | "B" | "C"
    eta_abs: float = 1.0          # доля поглощённой мощности (0..1)
    U_target: float = 0.9         # целевой вклад одного импульса (для режима "B")
    kappa_r: float = 0.5          # эффективный радиус ~ kappa_r * w0 (режим "C")
    kappa_z: float = 0.5          # эффективная глубина (режим "C")

    # --- Параметры источника мощности ---
    P_avg_W: float = 3.3                 # [Вт] — средняя мощность
    Pulsed: bool = True
    # Импульсные параметры (опциональны). Если заданы — использует source.py:
    pulse_duration_s: Optional[float] = None  # FWHM одного импульса, сек
    rep_rate_Hz: Optional[float] = None       # частота повторения, Гц
    E_pulse_J: Optional[float] = None         # энергия одного импульса, Дж
    pulse_count: Optional[int] = None         # количество импульсов в окне
    pulses_t0_s: float = 0.0                  # момент первого импульса в окне

    # --- Обучение/сэмплинг ---
    deltaT_scale: float = 1.0 # Параметр нормировки входных данных
    N_coll: int = 20000
    N_ic: int = 4096
    N_axis: int = 2048
    N_wall: int = 2048
    N_zbc: int = 1024
    source_calibration: str = 'Peak Power'

    widths: Tuple[int, ...] = (64, 64, 64, 64)
    lr: float = 1e-3

    w_pde: float = 1.0
    w_data: float = 1.0
    w_ic: float = 10.0
    w_bc: float = 10.0

    seed: int = 42
    device: str = "cpu"

    def validate(self):
        # Example checks
        # if self.pulse_count and self.rep_rate_Hz:
        #     total_time = self.pulse_count / self.rep_rate_Hz
        #     if total_time > self.Twindow_s:
        #         raise ValueError(
        #             f"Pulse train duration {total_time:.3e}s exceeds Twindow_s={self.Twindow_s:.3e}s"
        #             f"Pulse train duration {total_time:.3e}s exceeds Twindow_s={self.Twindow_s:.3e}s"
        #         )
        if self.P_avg_W is not None and self.P_avg_W < 0:
            raise ValueError("Average power must be non-negative")
