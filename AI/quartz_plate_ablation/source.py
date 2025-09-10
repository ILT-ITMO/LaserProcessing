# source.py
"""
Оптический источник для PINN с цилиндрической симметрией.
Читает параметры из пресетного JSON и возвращает безразмерный источник S(ρ, ζ, τ).

Нормировки:
  r = ρ * R_m, z = ζ * H_m, t = τ * Twindow_s
  T = T0_C + ΔT_scale * U  (U — предсказывает PINN)

Тогда безразмерный источник в уравнении для U:
  S(ρ,ζ,τ) = (Twindow_s / (rho*cp*ΔT_scale)) * q_W_m3(r,z,t)

Где q_W_m3 — объёмная мощность поглощения Гауссова пучка
с экспоненциальным затуханием по глубине (Beer–Lambert):
  q = μ_a * I_surf(r) * exp(-μ_a z),  μ_a = mu_star / H_m
  I_surf(r) = (2 P / (π w0^2)) * exp(-2 r^2 / w0^2)

Временной профиль в этой минимальной версии: g(τ)=1 на [0,1].
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import json
import torch
import torch.nn as nn

from config import Config                # dataclass-конфигурация
from pinn_io import load_params_json     # загрузка строго в Config


# ---------- Чтение параметров и "экстры" из JSON ----------

def load_cfg_and_extras(json_path: Path) -> Tuple[Config, Dict[str, Any]]:
    """
    Загружает Config и возвращает dict со ВСЕМИ полями JSON.
    Нужно, чтобы вытащить доп. поля (например, SCAN_SPEED_MM_S, DELTA_T_SCALE_K),
    которые не входят в Config.
    """
    json_path = Path(json_path)
    cfg = load_params_json(json_path)
    data = json.loads(json_path.read_text(encoding="utf-8"))
    return cfg, data


# ---------- Вспомогательное: ΔT_scale ----------

def require_deltaT_scale_from_json(data: Dict[str, Any]) -> float:
    """
    Обязательно достаём DELTA_T_SCALE_K из JSON.
    Если его нет — бросаем понятную ошибку: масштаб должен быть посчитан
    на этапе формирования пресета (create_preset_param.py).
    """
    if "DELTA_T_SCALE_K" not in data:
        raise KeyError(
            "В пресете отсутствует 'DELTA_T_SCALE_K'. "
            "Он должен быть посчитан в create_preset_param.py "
            "на основе temp_scaling_mode и сохранён в JSON."
        )
    try:
        val = float(data["DELTA_T_SCALE_K"])
    except Exception as e:
        raise ValueError("Поле 'DELTA_T_SCALE_K' должно быть числом (K на единицу U).") from e
    if not (val > 0.0):
        raise ValueError("Поле 'DELTA_T_SCALE_K' должно быть > 0.")
    return val


# ---------- Параметры источника ----------

@dataclass
class SourcePhys:
    P_W: float        # мощность, Вт
    w0_m: float       # радиус пучка по e^-2, м
    R_m: float        # радиус области, м
    H_m: float        # высота области, м
    Twindow_s: float  # временное окно, с
    mu_star: float    # H/δ  => δ = H/mu_star
    rho_kg_m3: float  # плотность, кг/м^3
    cp_J_kgK: float   # теплоёмкость, Дж/(кг·К)
    deltaT_scale_K: float  # масштаб ΔT для безразмерной температуры
    scan_speed_mm_s: Optional[float] = None  # на будущее (для бегущего пятна)

    @property
    def mu_abs_m_inv(self) -> float:
        # μ_a = mu_star / H
        return float(self.mu_star / self.H_m)


# ---------- Источник ----------

class OpticalSource(nn.Module):
    """
    S(ρ, ζ, τ) — безразмерный источник для уравнения на U(ρ, ζ, τ).
    """
    def __init__(self, params: SourcePhys, device: Optional[str] = None):
        super().__init__()
        self.p = params
        self.register_buffer("one", torch.tensor(1.0))
        if device is not None:
            self.to(device)

    @property
    def deltaT_scale_K(self) -> float:
        """Удобный доступ к масштабу ΔT (K на единицу U), проброшенному из пресета."""
        return float(self.p.deltaT_scale_K)

    # --- физический q(r,z,t) в Вт/м^3 ---
    def q_W_m3(self, rho: torch.Tensor, zeta: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        """
        q = μ_a * I_surf(r) * exp(-μ_a * z). Временной профиль top-hat: g(τ) = 1 на [0,1].
        """
        r = rho * self.p.R_m   # м
        z = zeta * self.p.H_m  # м

        # I(r) = 2P/(π w0^2) * exp(-2 r^2 / w0^2)
        I0 = 2.0 * self.p.P_W / (torch.pi * (self.p.w0_m ** 2))
        I_r = I0 * torch.exp(-2.0 * (r ** 2) / (self.p.w0_m ** 2))

        mu_a = self.p.mu_abs_m_inv
        q = mu_a * I_r * torch.exp(-mu_a * z)
        return q  # Вт/м^3

    # --- безразмерный S(ρ,ζ,τ) ---
    def forward(self, rho: torch.Tensor, zeta: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        """
        S = (Twindow_s / (ρ cp ΔT_scale)) * q_W_m3
        """
        q = self.q_W_m3(rho, zeta, tau)
        scale = self.p.Twindow_s / (self.p.rho_kg_m3 * self.p.cp_J_kgK * self.p.deltaT_scale_K)
        return q * scale


# ---------- Фабрика ----------

def build_optical_source(
    params_json_path: str | Path,
    *,
    deltaT_scale_override: Optional[float] = None,
    device: Optional[str] = None,
) -> OpticalSource:
    """
    Создаёт объект источника, используя пресетный JSON.
    Читает DELTA_T_SCALE_K, рассчитанный ранее в create_preset_param.py.
    При необходимости можно явно переопределить deltaT_scale_override.
    """
    cfg, data = load_cfg_and_extras(Path(params_json_path))

    if deltaT_scale_override is not None:
        deltaT_scale = float(deltaT_scale_override)
        if not (deltaT_scale > 0.0):
            raise ValueError("deltaT_scale_override должен быть > 0.")
    else:
        deltaT_scale = require_deltaT_scale_from_json(data)

    src = OpticalSource(
        SourcePhys(
            P_W=cfg.P_W,
            w0_m=cfg.w0_m,
            R_m=cfg.R_m,
            H_m=cfg.H_m,
            Twindow_s=cfg.Twindow_s,
            mu_star=cfg.mu_star,
            rho_kg_m3=cfg.rho_kg_m3,
            cp_J_kgK=cfg.cp_J_kgK,
            deltaT_scale_K=deltaT_scale,
            scan_speed_mm_s=data.get("SCAN_SPEED_MM_S", None),
        ),
        device=device or getattr(cfg, "device", None),
    )
    return src


# ---------- Утилита-сэмплер (опционально) ----------

@torch.no_grad()
def sample_collocation_from_cfg(cfg: Config, *, device: Optional[str] = None, N: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Простейший равномерный сэмплер (ρ, ζ, τ) ~ U(0,1), согласованный с cfg.
    Если N не указан — берём cfg.N_coll.
    Это лишь утилита; можно использовать ваш внешний скрипт сэмплинга.
    """
    device = device or getattr(cfg, "device", "cpu")
    N = int(N or cfg.N_coll)

    rho = torch.rand(N, 1, device=device)
    zeta = torch.rand(N, 1, device=device)
    tau = torch.rand(N, 1, device=device)
    return rho, zeta, tau


# ---------- Визуализация источника (опционально) ----------

def plot_source_2d(
    source: OpticalSource,
    *,
    tau_value: float = 0.5,
    Nr: int = 200,
    Nz: int = 200,
    device: Optional[str] = None,
    overlay_points: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    title: Optional[str] = None,
    show: bool = True,
    savepath: Optional[str | Path] = None,
):
    """
    Рисует карту S(ρ, ζ) при фиксированном τ (в безразмерных координатах).
    """
    import matplotlib.pyplot as plt

    device = device or next(source.parameters(), torch.tensor(0.)).device
    rho_lin = torch.linspace(0.0, 1.0, Nr, device=device)
    zeta_lin = torch.linspace(0.0, 1.0, Nz, device=device)
    RHO, ZETA = torch.meshgrid(rho_lin, zeta_lin, indexing="ij")
    TAU = torch.full_like(RHO, float(tau_value))

    with torch.no_grad():
        S = source(RHO, ZETA, TAU).detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(6.2, 4.8), dpi=120)
    im = ax.imshow(S.T, origin="lower", extent=[0, 1, 0, 1], aspect="auto")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("S (безразмерный)")
    ax.set_xlabel("ρ")
    ax.set_ylabel("ζ")
    ax.set_title(title or f"S(ρ, ζ) при τ = {tau_value:.2f}")

    if overlay_points is not None:
        rho_pts, zeta_pts = overlay_points
        rp = rho_pts.detach().flatten().cpu().numpy()
        zp = zeta_pts.detach().flatten().cpu().numpy()
        ax.plot(rp, zp, "o", ms=2.5, mfc="none")

    fig.tight_layout()
    if savepath is not None:
        fig.savefig(savepath, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_source_2d_physical(
    source: OpticalSource,
    *,
    tau_value: float = 0.5,
    Nr: int = 200,
    Nz: int = 200,
    r_unit: str = "mm",     # "m" | "mm" | "um"
    z_unit: str = "um",     # "m" | "mm" | "um"
    quantity: str = "q",    # "q" -> Вт/м^3, "S" -> безразмерный
    device: Optional[str] = None,
    overlay_points: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # (rho_pts, zeta_pts) в [0,1]
    title: Optional[str] = None,
    show: bool = True,
    savepath: Optional[str | Path] = None,
):
    """
    Рисует 2D-карту источника в ФИЗИЧЕСКИХ координатах r–z.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # факторы перевода осей
    unit_factor = {"m": 1.0, "mm": 1e3, "um": 1e6}
    if r_unit not in unit_factor or z_unit not in unit_factor:
        raise ValueError("r_unit/z_unit must be one of: 'm', 'mm', 'um'")

    p = source.p  # SourcePhys
    r_scale = unit_factor[r_unit]
    z_scale = unit_factor[z_unit]

    device = device or next(source.parameters(), torch.tensor(0.)).device
    # нормированные сетки
    rho_lin = torch.linspace(0.0, 1.0, Nr, device=device)
    zeta_lin = torch.linspace(0.0, 1.0, Nz, device=device)
    RHO, ZETA = torch.meshgrid(rho_lin, zeta_lin, indexing="ij")
    TAU = torch.full_like(RHO, float(tau_value))

    # считаем поле
    with torch.no_grad():
        if quantity.lower() == "q":
            VAL = source.q_W_m3(RHO, ZETA, TAU)              # Вт/м^3
            cbar_label = "q (Вт/м³)"
        elif quantity.lower() == "s":
            VAL = source(RHO, ZETA, TAU)                     # безразмерный
            cbar_label = "S (безразмерный)"
        else:
            raise ValueError("quantity must be 'q' or 'S'")

        VAL = VAL.detach().cpu().numpy()

    # физические пределы осей в выбранных единицах
    r_max = p.R_m * r_scale
    z_max = p.H_m * z_scale

    fig, ax = plt.subplots(figsize=(6.6, 4.8), dpi=120)
    im = ax.imshow(
        VAL.T,
        origin="lower",
        extent=[0.0, r_max, 0.0, z_max],  # r, z
        aspect="auto",
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)
    ax.set_xlabel(f"r, {r_unit}")
    ax.set_ylabel(f"z, {z_unit}")
    t_us = float(tau_value) * source.p.Twindow_s * 1e6  # реальное время, мкс
    name = "q" if quantity.lower() == "q" else "S"
    ax.set_title(title or f"{name}(r, z) при τ = {tau_value:.2f}  (t = {t_us:.2f} µs)")

    # наложение точек (ρ, ζ) -> (r, z) в выбранных единицах
    if overlay_points is not None:
        rho_pts, zeta_pts = overlay_points
        r_pts = (rho_pts.detach().flatten().cpu().numpy() * p.R_m) * r_scale
        z_pts = (zeta_pts.detach().flatten().cpu().numpy() * p.H_m) * z_scale
        ax.plot(r_pts, z_pts, "o", ms=2.5, mfc="none")

    fig.tight_layout()
    if savepath is not None:
        fig.savefig(savepath, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    # Пример самопроверки: замените путь на актуальный ваш JSON
    PARAMS_JSON = Path("presets_params/pinn_params_P3p3W_V60mms_20250908_172037.json")
    source = build_optical_source(PARAMS_JSON)

    # Пример: наложим коллокационные точки
    cfg, _ = load_cfg_and_extras(PARAMS_JSON)
    rho_pts, zeta_pts, tau_pts = sample_collocation_from_cfg(cfg, N=min(1000, getattr(cfg, "N_coll", 1000)))
    plot_source_2d_physical(
        source,
        tau_value=0.5,
        overlay_points=(rho_pts, zeta_pts),
        savepath=None,  # например: "source_map.png"
    )
