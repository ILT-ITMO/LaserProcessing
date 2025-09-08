# ===============================
# pinn_heat/config.py
# ===============================
from dataclasses import dataclass
from typing import Tuple

@dataclass
class Config:
    # --- Геометрия и нормировка ---
    R_m: float              # Радиус области моделирования (метры)
    H_m: float              # Высота области моделирования (метры)
    w0_m: float             # Радиус лазерного пучка по e^-2 (метры)
    Twindow_s: float        # Временное окно моделирования (секунды)
    mu_star: float          # Безразмерная глубина поглощения (H/δ), где δ — длина затухания
    T0_C: float             # Начальная температура образца (°C)

    # --- Мощность и материал ---
    P_W: float              # Оптическая мощность лазера (Вт)
    rho_kg_m3: float        # Плотность материала (кг/м^3)
    cp_J_kgK: float         # Удельная теплоёмкость (Дж/(кг·K))
    k_W_mK: float           # Теплопроводность (Вт/(м·K))

    # --- Параметры сэмплинга точек ---
    N_coll: int = 20000     # Количество коллокационных точек для PDE-резидуала
    N_ic: int = 4096        # Точек для начальных условий (t=0)
    N_axis: int = 2048      # Точек для граничного условия на оси (ρ=0)
    N_wall: int = 2048      # Точек для граничного условия на стенке (ρ=1)
    N_zbc: int = 1024       # Точек для граничных условий по глубине (ζ=0, ζ=1)

    # --- Гиперпараметры модели и обучения ---
    widths: Tuple[int, ...] = (64, 64, 64, 64)   # Размерности скрытых слоёв MLP
    lr: float = 1e-3                             # Скорость обучения оптимизатора
    w_pde: float = 1.0                           # Вес лосса PDE
    w_data: float = 1.0                          # Вес лосса на экспериментальных данных (если есть)
    w_ic: float = 10.0                           # Вес лосса на начальном условии
    w_bc: float = 10.0                           # Вес лосса на граничных условиях
    seed: int = 42                               # Сид генератора случайных чисел
    device: str = "cuda"                         # Устройство для вычислений ("cuda" или "cpu")

    # def deltaT_scale_K(self) -> float:
    #     """Compute ΔT_scale based on the same formula as in your current code.
    #     For now we try to reuse the implementation from axial_symmetry_problem if present.
    #     """
    #     try:
    #         from axial_symmetry_problem import compute_deltaT_scale as _f
    #         return float(_f(self.P_W, self.w0_m, self.rho_kg_m3, self.cp_J_kgK, self.mu_star, self.Twindow_s))
    #     except Exception:
    #         # Minimal placeholder (so code still runs). Replace with real formula during migration.
    #         return 1.0

    def validate(self) -> None:
        # Проверки корректности параметров
        assert self.R_m > 0 and self.H_m > 0 and self.w0_m > 0
        assert self.P_W >= 0 and self.rho_kg_m3 > 0 and self.cp_J_kgK > 0
        assert self.Twindow_s > 0 and self.mu_star > 0
