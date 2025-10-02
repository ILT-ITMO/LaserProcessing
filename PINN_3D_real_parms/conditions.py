from __future__ import annotations
import torch
import config
from normalizer import QuartzNormalizer
import physical_params as phys
import matplotlib.pyplot as plt
import torch
import numpy as np

import math
import torch.nn as nn

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
# Используем нормализатор из config
# normalizer = config.normalizer

def compute_centers(n, spacing, device):
    if n % 2 == 1:
        idx = torch.arange(-(n//2), n//2 + 1, device=device, dtype=torch.float32)
    else:
        half = n // 2
        idx = torch.arange(-half + 0.5, half, step=1.0, device=device, dtype=torch.float32)

    centers = []
    for i in idx:
        for j in idx:
            for k in idx:
                centers.append(torch.tensor([i * spacing, j * spacing, k * spacing], device=device))
    return centers

def initial_gaussian(x_tensor, y_tensor, z_tensor, t0=1.0):
    """Начальное условие в безразмерных координатах"""
    centers = compute_centers(phys.NUM_GAUSSIANS, phys.GAUSSIAN_SPACING, x_tensor.device)
    gaussians = []
    for c in centers:
        r_squared = (x_tensor - c[0])**2 + (y_tensor - c[1])**2 + (z_tensor - c[2])**2
        gaussians.append(t0 * torch.exp(-r_squared / (2 * phys.SIGMA0**2)))
    return sum(gaussians)

def laser_source_term(x_tensor, y_tensor, z_tensor, t_tensor, H_z, MU):
    """
    ПРАВИЛЬНАЯ функция лазерного источника тепла согласно физике
    Соответствует закону Бугера-Ламберта для экспоненциального поглощения
    """
    MU_STAR = H_z * MU

    # Центрируем координаты (пучок в центре области)
    x_centered = x_tensor - 0.5  
    y_centered = y_tensor - 0.5
    
    spatial_dist = torch.exp(-(x_centered**2 + y_centered**2) / (2 * phys.LASER_SIGMA**2))
    
    DUR = phys.LASER_PULSE_PERIOD / 500*10**(-6)

    t_mod = torch.fmod(t_tensor, DUR) # laser_pulse_period / T_MAX
    # Центр импульса в середине длительности
    pulse_center = DUR / 2
    temporal_dist = torch.exp(-(t_mod - pulse_center)**2 / (2 * (DUR/4)**2))
    
    depth_dist = MU_STAR * torch.exp(-MU_STAR * z_tensor)
    
    # Нормируем временной профиль чтобы интеграл = 1
    # temporal_norm = 1.0 / (config.LASER_PULSE_DURATION * torch.sqrt(torch.tensor(2 * torch.pi)))
    # temporal_dist = temporal_dist * temporal_norm
    
    return (phys.I0 * spatial_dist * temporal_dist * depth_dist, temporal_dist)






# import numpy as np
# import physical_params
# import matplotlib.pyplot as plt

# device = 'cpu'
# x_phys = np.linspace(physical_params.X_MIN, physical_params.X_MAX, 20)  # м
# y_phys = np.linspace(physical_params.Y_MIN, physical_params.Y_MAX, 20)  # м
# z_phys = np.linspace(physical_params.Z_MIN, physical_params.Z_MAX, 20)  # м
# t_phys = np.linspace(0, physical_params.T_MAX, 20) 

# (x_norm, x_coef), (y_norm, y_coef), (z_norm, z_coef), (t_norm, t_coef) = QuartzNormalizer.normalize_vector(x_phys, y_phys, z_phys, t_phys, get_coef=True)


# X, Y, Z, T = torch.meshgrid(x_norm, y_norm, z_norm, t_norm, indexing='ij')
# x_coll = X.flatten()
# y_coll = Y.flatten()
# z_coll = Z.flatten()
# t_coll = T.flatten()

# X_c, Y_c, Z_c, T_c = torch.meshgrid(x_coef, y_coef, z_coef, t_coef, indexing='ij')
# x_coef = X.flatten()
# y_coef = Y.flatten()
# z_coef = Z.flatten()
# t_coef = T.flatten()


# # print(z_norm.shape)
# # print(z_coef.shape)
# checker = laser_source_term(x_coll, y_coll, z_coll, t_coll, z_coef, physical_params.MU)
# print(checker)


# plt.figure(figsize=(10, 4))
# plt.plot(t_coll.numpy(), checker[1].numpy())
# plt.xlabel('Время (с)')
# plt.ylabel('Интенсивность (Вт/м³)')
# plt.title('Временной профиль лазерного импульса')
# plt.grid(True)
# plt.show()




# ---------- Параметры источника ----------

@dataclass
class SourcePhys:
    # Геометрия/материал/нормировки
    w0_m: float
    R_m: float
    H_m: float
    Twindow_s: float
    mu_star: float
    rho_kg_m3: float
    cp_J_kgK: float
    deltaT_scale_K: float
    # Временные параметры
    pulse_fwhm_s: Optional[float] = None
    rep_rate_Hz: Optional[float] = None
    pulse_count: Optional[int] = None
    pulses_t0_s: float = 0.0
    P_avg_W: Optional[float] = None
    P_peak_W: Optional[float] = None
    # Прочее
    @property
    def mu_abs_m_inv(self) -> float:
        # μ_a = mu_star / H
        return float(self.mu_star / self.H_m)

    @property
    def gaussian_area_coeff(self) -> float:
        # ∫ exp(-4 ln2 * t^2 / FWHM^2) dt = FWHM * sqrt(pi/(4 ln 2))
        if self.pulse_fwhm_s is None:
            return 1.0
        return self.pulse_fwhm_s * math.sqrt(math.pi / (4.0 * math.log(2.0)))

# ---------- Источник ----------
class OpticalSource(nn.Module):
    """
    S(x, y, ζ, τ) — безразмерный источник для уравнения на U(x, y, ζ, τ)
    c временной огибающей как гребёнка гауссовых импульсов.
    """
    def __init__(self, params: SourcePhys, device: Optional[str] = None):
        super().__init__()
        self.p = params
        # предрасчёт временных центров импульсов (в секундах)
        self._pulse_centers_s: List[float] = self._compute_pulse_centers()
        self.register_buffer("one", torch.tensor(1.0))
        if device is not None:
            self.to(device)
        # Перевод в пиковую мощность
        self._infer_peak_power_from_average()

    # --------- служебные расчёты времени/мощности ----------

    def _compute_pulse_centers(self) -> List[float]:
        p = self.p
        t0 = float(p.pulses_t0_s or 0.0)
        Tw = float(p.Twindow_s)
        # Приоритет: заданный pulse_count; иначе из rep_rate*window; иначе одиночный
        if p.rep_rate_Hz and p.rep_rate_Hz > 0.0:
            if p.pulse_count is None:
                count = int(max(0, math.floor(p.rep_rate_Hz * Tw)))
            else:
                count = int(max(0, p.pulse_count))
            centers = [t0 + k / p.rep_rate_Hz for k in range(count)]
        else:
            # без частоты: одиночный импульс, центр в середине окна
            centers = [t0 if (0.0 <= t0 <= Tw) else 0.5 * Tw]

        # Оставим только те, что попадают в окно
        return [t for t in centers if (0.0 <= t <= Tw)]

    def _infer_peak_power_from_average(self) -> None:
        """
        Если P_peak_W не задана, но есть средняя мощность (P_avg_W или P_W из cfg),
        и заданы rep_rate_Hz и pulse_fwhm_s, то восстановим пиковую мощность.
        """
        p = self.p
        if p.P_peak_W is not None:
            return
        if (p.P_avg_W is None) or (p.pulse_fwhm_s is None) or (p.rep_rate_Hz is None) or (p.rep_rate_Hz <= 0.0):
            return
        area = p.gaussian_area_coeff  # FWHM*sqrt(pi/(4 ln2))
        if area <= 0.0:
            return
        # P_avg = P_peak * area * f_rep  =>  P_peak = P_avg / (area * f_rep)
        P_peak = float(p.P_avg_W) / (area * float(p.rep_rate_Hz))
        self.p.P_peak_W = max(P_peak, 0.0)

    # --------- временная огибающая (торч) ----------

    def temporal_envelope(self, tau: torch.Tensor) -> torch.Tensor:
        """
        Возвращает Σ_k exp(-4 ln2 * (t - t_k)^2 / FWHM^2) в момент времени t = tau*Twindow.
        Если FWHM или центры не определены — возвращает единицу (CW).
        """
        if (self.p.pulse_fwhm_s is None) or (len(self._pulse_centers_s) == 0):
            return torch.ones_like(tau)

        t = tau * self.p.Twindow_s  # сек
        fwhm = float(self.p.pulse_fwhm_s)
        c = -4.0 * math.log(2.0) / (fwhm * fwhm)

        # Сумма гауссов по центрам. Для численной устойчивости — складываем в торче.
        env = torch.zeros_like(tau)
        for tk in self._pulse_centers_s:
            env = env + torch.exp(c * (t - tk) ** 2)
        return env

    @property
    def deltaT_scale_K(self) -> float:
        """Удобный доступ к масштабу ΔT (K на единицу U), проброшенному из пресета."""
        return float(self.p.deltaT_scale_K)

    # --- физический q(x,y,z,t) в Вт/м^3 ---
    def q_W_m3(self, x: torch.Tensor, y: torch.Tensor, zeta: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        """
        q = μ_a * I_surf(x,y,t) * exp(-μ_a * z),
        I_surf(x,y,t) = (2 P(t) / (π w0^2)) * exp(-2 (x^2 + y^2) / w0^2)
        """
        # Преобразование безразмерных координат в физические (метры)
        x_phys = x * self.p.R_m   # м
        y_phys = y * self.p.R_m   # м  
        z = zeta * self.p.H_m     # м

        # Вычислим мгновенную мощность P(t)
        env = self.temporal_envelope(tau)  # сумма гауссов без масштаба
        # Масштаб мощности: используем пиковую мощность; если её нет — fallback к P_avg или 0.
        if self.p.P_peak_W is not None:
            P_t = float(self.p.P_peak_W) * env
        elif self.p.P_avg_W is not None:
            # если нет rep_rate/FWHM, env≈1 => трактуем как CW со средней мощностью
            P_t = float(self.p.P_avg_W) * env
        else:
            # финальный запасной вариант — CW с cfg.P_W (может быть 0)
            P_t = float(getattr(self.p, "P_avg_W", 0.0)) * env  # обычно недостижимо

        # I(x,y,t) = 2P(t)/(π w0^2) * exp(-2 (x^2 + y^2) / w0^2)
        I0 = 2.0 / (math.pi * (self.p.w0_m ** 2))
        r_squared = x_phys ** 2 + y_phys ** 2
        I_xy_t = I0 * P_t * torch.exp(-2.0 * r_squared / (self.p.w0_m ** 2))

        mu_a = self.p.mu_abs_m_inv
        q = mu_a * I_xy_t * torch.exp(-mu_a * z)
        return q  # Вт/м^3

    # --- безразмерный S(x,y,ζ,τ) ---
    def forward(self, x: torch.Tensor, y: torch.Tensor, zeta: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        """
        S = (Twindow_s / (ρ cp ΔT_scale)) * q_W_m3
        """
        q = self.q_W_m3(x, y, zeta, tau)
        scale = self.p.Twindow_s / (self.p.rho_kg_m3 * self.p.cp_J_kgK * self.deltaT_scale_K)
        return q * scale
    

# Создаем источник с тестовыми параметрами
params = SourcePhys(
    w0_m=62e-6,      # радиус пучка
    R_m=300e-6,        # радиус области
    H_m=100e-6,       # толщина
    Twindow_s=0.0005,# временное окно 500 мкс
    mu_star=10,    # коэффициент поглощения
    rho_kg_m3=2200.0,  # плотность
    cp_J_kgK=740.0,   # теплоемкость
    deltaT_scale_K=1.0,
    pulse_fwhm_s=0.000015,  # длительность импульса 15 мкс
    rep_rate_Hz=8000,     # частота 8 кГц
    P_avg_W= 3.3       
)

# source = OpticalSource(params)

# # Временные точки для графика
# tau = torch.linspace(0, 1, 1000)  # безразмерное время
# x_center = torch.tensor(0.0)      # центр по x
# y_center = torch.tensor(0.0)      # центр по y  
# zeta_surface = torch.tensor(0.0)  # поверхность (z=0)

# # Вычисляем интенсивность источника во времени
# intensity = source(x_center, y_center, zeta_surface, tau)

# # Строим график
# plt.figure(figsize=(12, 6))
# plt.plot(tau.numpy() * params.Twindow_s * 1e6, intensity.detach().numpy(), 'b-', linewidth=2)
# plt.xlabel('Время (мкс)')
# plt.ylabel('Интенсивность источника (безразм.)')
# plt.title('Временная зависимость интенсивности лазерного источника')
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.show()