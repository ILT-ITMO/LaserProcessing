from dataclasses import dataclass, asdict
from typing import Callable, Dict, Any, Optional, Union
import numpy as np

ArrayLike = Union[float, int, np.ndarray, list, tuple]

@dataclass
class CylScales:
    R: float          # радиус расчетной области [m]
    H: float          # глубина [m]
    T: float          # временное окно [s]
    w0: float         # лучевой радиус пучка [m]
    w0_star: float    # безразмерный лучевой радиус (w0/R)
    alpha: float      # тепловая диффузивность [m^2/s]
    beta_r: float     # α T / R^2
    beta_z: float     # α T / H^2
    beta_single: float  # (β_r + β_z)/2
    P_scale: float    # масштаб мощности [W] (для нормировки параметров источника)
    k: Optional[float] = None
    rho: Optional[float] = None
    cp: Optional[float] = None

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

class CylindricalNormalizer:
    """
    Нормировка для осесимметричной задачи теплопереноса с гауссовым пучком.
    Координаты: r∈[0,R] -> ρ∈[0,1], z∈[0,H] -> ζ∈[0,1], t∈[0,T] -> τ∈[0,1].
    PDE в нормированных переменных:
        T_τ = β_r (T_ρρ + (1/ρ) T_ρ) + β_z T_ζζ + S'(ρ,ζ,τ),  S' = T S.
    По желанию можно использовать единый β = (β_r+β_z)/2.
    """
    def __init__(
        self,
        R_m: float, H_m: float, T_s: float,
        w0_m: float,
        *,
        # Параметры источника (необязательно, но удобно иметь под рукой):
        P_W: Optional[float] = None,
        # Термофизика:
        alpha_m2_s: Optional[float] = None,
        k_W_mK: Optional[float] = None, rho_kg_m3: Optional[float] = None, cp_J_kgK: Optional[float] = None,
        # Масштаб мощности (если хотите P* = 1 — укажите P_scale_override_W=P_W)
        P_scale_override_W: Optional[float] = None,
    ):
        self.R, self.H, self.T = float(R_m), float(H_m), float(T_s)
        self.w0 = float(w0_m)
        self.P = None if P_W is None else float(P_W)

        # alpha напрямую или из (k, rho, cp)
        if alpha_m2_s is not None:
            alpha = float(alpha_m2_s)
        elif (k_W_mK is not None) and (rho_kg_m3 is not None) and (cp_J_kgK is not None):
            alpha = float(k_W_mK) / (float(rho_kg_m3) * float(cp_J_kgK))
        else:
            raise ValueError("Укажите либо alpha_m2_s, либо (k, rho, cp).")

        beta_r = alpha * self.T / (self.R**2)
        beta_z = alpha * self.T / (self.H**2)
        beta_single = 0.5 * (beta_r + beta_z)

        w0_star = self.w0 / self.R

        if P_scale_override_W is not None:
            P_scale = float(P_scale_override_W)
        else:
            # безопасный дефолт: масштаб = P (если задана), иначе 1 Вт
            P_scale = self.P if (self.P is not None and self.P > 0) else 1.0

        self.scales = CylScales(
            R=self.R, H=self.H, T=self.T,
            w0=self.w0, w0_star=w0_star,
            alpha=alpha, beta_r=beta_r, beta_z=beta_z, beta_single=beta_single,
            P_scale=P_scale, k=k_W_mK, rho=rho_kg_m3, cp=cp_J_kgK
        )

        # аффинные карты
        self.fwd_r: Callable[[ArrayLike], np.ndarray]  = lambda r: np.asarray(r, dtype=float) / self.R      # ρ
        self.inv_r: Callable[[ArrayLike], np.ndarray]  = lambda rho: np.asarray(rho, dtype=float) * self.R   # r
        self.fwd_z: Callable[[ArrayLike], np.ndarray]  = lambda z: np.asarray(z, dtype=float) / self.H      # ζ
        self.inv_z: Callable[[ArrayLike], np.ndarray]  = lambda zt: np.asarray(zt, dtype=float) * self.H    # z
        self.fwd_t: Callable[[ArrayLike], np.ndarray]  = lambda t: np.asarray(t, dtype=float) / self.T      # τ
        self.inv_t: Callable[[ArrayLike], np.ndarray]  = lambda tau: np.asarray(tau, dtype=float) * self.T  # t

    # ------- Выдача безразмерных параметров -------
    def params(self) -> Dict[str, Any]:
        s = self.scales
        out = {
            "rho_of_r": "rho = r / R ∈ [0,1]",
            "zeta_of_z": "zeta = z / H ∈ [0,1]",
            "tau_of_t": "tau = t / T ∈ [0,1]",
            "w0_star": s.w0_star,
            "beta_r": s.beta_r,
            "beta_z": s.beta_z,
            "beta_single": s.beta_single,
            "P_star": (None if self.P is None else self.P / s.P_scale),
            "scales": s.as_dict(),
        }
        return out

    # ------- Гауссов профиль и мощность в безразмерном виде -------
    def gaussian_profile(self, rho: ArrayLike) -> np.ndarray:
        """Безразмерный радиальный профиль G(ρ) = exp( -2 (ρ / w0*)^2 )."""
        rho = np.asarray(rho, dtype=float)
        w0s = self.scales.w0_star
        return np.exp(-2.0 * (rho / w0s) ** 2)

    def power_dim2star(self, P_W: ArrayLike) -> np.ndarray:
        """Нормировка мощности: P* = P / P_scale."""
        return np.asarray(P_W, dtype=float) / self.scales.P_scale

    def power_star2dim(self, P_star: ArrayLike) -> np.ndarray:
        """Обратная нормировка мощности."""
        return np.asarray(P_star, dtype=float) * self.scales.P_scale

if __name__ == "__main__":
    # Радиальный максимум разумно взять как половину ширины канала (ось симметрии в центре):
    W = 152.4e-6     # ширина канала
    R = W/2          # радиус расчетной области
    H = 21.9e-6      # глубина
    Twindow = 50e-3  # окно времени, которое маппится в [0,1]
    w0 = 31.0e-6     # лучевой радиус пучка (для пятна 62 μm по 1/e)
    P = 11.1         # Вт (если нужно нормировать параметр мощности)

    # Теплофизика кварца (примерно):
    k, rho, cp = 1.38, 2200.0, 703.0
    alpha = k/(rho*cp)  # ~8.9e-7 м^2/с

    norm = CylindricalNormalizer(
        R_m=R, H_m=H, T_s=Twindow, w0_m=w0,
        P_W=P, alpha_m2_s=alpha,
        P_scale_override_W=P  # хотим P* = 1 для этого опыта
    )
    par = norm.params()
    print("w0* =", par["w0_star"])
    print("β_r =", par["beta_r"], "  β_z =", par["beta_z"], "  β(single) =", par["beta_single"])

    # Пример: радиальный профиль источника в безразмерных координатах:
    rho_line = np.linspace(0, 1, 5)
    print("G(rho) =", norm.gaussian_profile(rho_line))

