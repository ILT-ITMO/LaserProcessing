# visualize_pinn_field.py
from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from config import Config
from source import load_cfg_and_extras, build_optical_source

# ======== CONFIG ========
PRESET_PATH = Path("./presets_params/pinn_params_P3p3W_V40mms_20250911_164958.json")
CKPT_PATH   = Path("checkpoints/best.pt")
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
SEED        = 123

NR, NZ = 256, 256               # сетка по ρ и ζ
TAU_LIST = [0.0, 0.1, 0.15, 0.20, 0.25, 0.3, 0.6]
# TAU_LIST = [0.95, 0.96, 0.97, 0.98,0.99, 1.0]   # моменты времени (τ in [0,1])

CMAP = "inferno"
VMIN, VMAX = None, None         # лимиты цветовой шкалы по T(°C); None = авто-глобальные

ISO_TEMP_C: Optional[float] = None        # сплошная красная изотерма
ISO_DASH_TEMP_C: Optional[float] = None # пунктирная изотерма; например 200.0
ISO_DASH_TEMP_C = 50

ISO_SNAP_TO_RANGE = True                  # «прилипать» изотерме к границе диапазона кадра

# ==== OVERLAY: experimental groove profile ====
PROFILE_OVERLAY_ENABLE = True
PROFILE_OVERLAY_PATH: Optional[Path] = Path("fitted_profile_for_pinn.npz")
# X-ось берём как радиус r: поддерживаются r_um/x_um (μм), r_m/x_m (м), rho (0..1)
PROFILE_OVERLAY_X_AXIS = "auto"   # "auto" | "r_um" | "x_um" | "r_m" | "x_m" | "rho"
# В NPZ y_fit трактуем как ВЫСОТУ (не глубину); глубину считаем как (baseline - y)
PROFILE_Y_AS_DEPTH = False
PROFILE_UNIT_Y_TO_M = 1e-6        # если y в μм → 1e-6 (м/μм)
PROFILE_Z_OFFSET_UM = 0.0         # вертикальный сдвиг кривой (μм)
PROFILE_STYLE = dict(color="cyan", linewidth=2.0, alpha=0.95, label="эксп. профиль")
# ==============================================

OUT_DIR = Path("viz")
SAVE_NPZ = True
# ========================

def _get(obj: Any, name: str, default=None):
    return getattr(obj, name, default) if hasattr(obj, name) else (obj.get(name, default) if isinstance(obj, dict) else default)

def build_model_from_cfg(cfg: Config) -> nn.Module:
    import inspect, models
    if hasattr(models, "build_model") and callable(models.build_model):
        return models.build_model(cfg)  # type: ignore
    model_name = str(_get(cfg, "model_name", "")).strip()
    if model_name:
        cand = getattr(models, model_name, None)
        if inspect.isclass(cand) and issubclass(cand, nn.Module):
            return cand(cfg) if "cfg" in inspect.signature(cand).parameters else cand()
    for name in ("PINNModel", "PINN", "MLP"):
        cand = getattr(models, name, None)
        if inspect.isclass(cand) and issubclass(cand, nn.Module):
            return cand(cfg) if "cfg" in inspect.signature(cand).parameters else cand()
    for name, obj in vars(models).items():
        if inspect.isclass(obj) and issubclass(obj, nn.Module):
            try:
                return obj(cfg) if "cfg" in inspect.signature(obj).parameters else obj()
            except Exception:
                continue
    raise RuntimeError("Не удалось построить модель из models.py")

def resolve_deltaT_scale_K(cfg: Config, extras: Dict[str, Any], src_obj: Any) -> float:
    if isinstance(extras, dict) and "DELTA_T_SCALE_K" in extras:
        try:
            val = float(extras["DELTA_T_SCALE_K"]);  assert val > 0
            return val
        except Exception:
            pass
    val = getattr(src_obj, "deltaT_scale_K", None)
    if val is not None:
        try:
            v = float(val);  assert v > 0
            return v
        except Exception:
            pass
    for name in ("dT_per_U_C", "U_to_dT_C_scale"):
        v = getattr(cfg, name, None)
        if v is not None:
            try:
                v = float(v);  assert v > 0
                return v
            except Exception:
                pass
    raise RuntimeError("Не найден ΔT_scale (К/ед. U) для перевода U→°C.")

def load_checkpoint_into(model: nn.Module, ckpt_path: Path, device: torch.device) -> None:
    if not ckpt_path.exists():
        print(f"[WARN] Нет файла чекпоинта: {ckpt_path}. Использую случайную инициализацию.")
        return
    try:
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        state = torch.load(ckpt_path, map_location=device)
    model_state = state.get("model_state", state) if isinstance(state, dict) else state
    model.load_state_dict(model_state)

def eval_slice_fixed(model: nn.Module, tau: float, RHO: np.ndarray, ZETA: np.ndarray, device: torch.device) -> np.ndarray:
    nr = RHO.shape[1];  nz = ZETA.shape[0]
    rho_t = torch.from_numpy(RHO.reshape(-1, 1)).to(device)
    zeta_t = torch.from_numpy(ZETA.reshape(-1, 1)).to(device)
    tau_t = torch.full((rho_t.shape[0], 1), float(tau), dtype=rho_t.dtype, device=device)
    with torch.no_grad():
        U = model(rho_t, zeta_t, tau_t)
    return U.detach().cpu().numpy().reshape(nz, nr)

# -------- Изотермы --------
def draw_iso_contour(
    ax,
    x_um: np.ndarray,
    y_um: np.ndarray,
    T: np.ndarray,
    temp_C: float,
    *,
    color: str = "red",
    linestyle: str = "-",
    linewidth: float = 1.25,
    label: bool = True,
    label_fmt = None,
    snap_to_range: bool = False,
):
    if label_fmt is None:
        label_fmt = lambda v: f"{v:.0f} °C"
    tmin = float(np.nanmin(T));  tmax = float(np.nanmax(T))
    eps = max(1e-6, 1e-3 * (tmax - tmin + 1e-12))
    if temp_C < tmin - eps or temp_C > tmax + eps:
        if not snap_to_range:
            msg = f"No iso: {temp_C:.2f} °C (range [{tmin:.2f}, {tmax:.2f}] °C)"
            ax.text(0.01, 0.99, msg, transform=ax.transAxes, va='top', ha='left',
                    fontsize=8, color=color, bbox=dict(facecolor='white', alpha=0.55, edgecolor='none'))
            print("[ISO WARN]", msg)
            return None
        temp_C = min(max(temp_C, tmin), tmax)
        ax.text(0.01, 0.99, f"ISO snapped→{temp_C:.2f} °C\n[{tmin:.2f},{tmax:.2f}]",
                transform=ax.transAxes, va='top', ha='left', fontsize=8, color=color,
                bbox=dict(facecolor='white', alpha=0.55, edgecolor='none'))
    cs = ax.contour(x_um, y_um, T, levels=[float(temp_C)],
                    colors=color, linestyles=linestyle, linewidths=linewidth, zorder=10)
    if label:
        ax.clabel(cs, fmt=label_fmt, inline=True, fontsize=8)
    return cs

# ==== OVERLAY helpers ====
def _orient_first_point_at_axis_z(r_um: np.ndarray, z_um: np.ndarray, *, r_zero_tol_um: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Сделать первую точку самой близкой к оси z (минимальный r), при необходимости защёлкнуть на r=0."""
    if r_um.size == 0:
        return r_um, z_um
    idx = int(np.nanargmin(r_um))
    if idx != 0:
        r_um = np.concatenate([r_um[idx:], r_um[:idx]])
        z_um = np.concatenate([z_um[idx:], z_um[:idx]])
    # монотонность по r
    if np.nanmedian(np.diff(r_um)) < 0:
        r_um = r_um[::-1];  z_um = z_um[::-1]
    if r_um[0] <= max(0.0, float(r_zero_tol_um)):
        r_um[0] = 0.0
    return r_um, z_um

def _load_overlay_profile_npz(
    path: Path,
    cfg: Config,
    *,
    prefer_axis: str = "auto",
    y_is_depth: bool = False,
    unit_y_to_m: float = 1e-6
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Возвращает (r_um, z_um) для наложения.
    - r_um: горизонталь (μм) от 0 до R
    - z_um: глубина (μм) положительная вниз (как на тепловой карте)
    Интерпретация: X — радиус r; Y — высота → глубина = baseline - y.
    """
    if not path.exists():
        raise FileNotFoundError(f"Файл профиля не найден: {path}")
    with np.load(path, allow_pickle=True) as data:
        # X-ось
        key = None
        if prefer_axis != "auto" and prefer_axis in data:
            key = prefer_axis
        else:
            for k in ("r_um", "x_um", "r_m", "x_m", "rho"):
                if k in data:
                    key = k;  break
        if key is None:
            raise KeyError("В NPZ нет ни 'r_um', 'x_um', 'r_m', 'x_m', ни 'rho'.")

        if key in ("r_um", "x_um"):
            r_m = np.asarray(data[key], float) * 1e-6         # μм → м
        elif key in ("r_m", "x_m"):
            r_m = np.asarray(data[key], float)                # м
        elif key == "rho":
            R_m = float(getattr(cfg, "R_m", 1.0))
            r_m = np.asarray(data["rho"], float) * R_m
        else:
            raise KeyError(f"Неизвестный ключ X: {key}")

        # Y-профиль (высота или глубина)
        y = np.asarray(data["y_fit"] if "y_fit" in data else data["y"], float)

        if y_is_depth:
            depth_m = y * float(unit_y_to_m)
        else:
            # y — высота поверхности; считаем положительную глубину вниз
            baseline = None
            try:
                if "params" in data:
                    params = data["params"].item()
                    baseline = float(params.get("baseline", np.nan))
            except Exception:
                baseline = None
            if baseline is None or not np.isfinite(baseline):
                baseline = float(np.nanmax(y))
            depth_m = (baseline - y) * float(unit_y_to_m)

    # Гарантируем положительное направление глубины
    if np.nanmedian(depth_m) < 0:
        depth_m = -depth_m

    r_um = r_m * 1e6
    z_um = depth_m * 1e6

    # фильтрация и ориентация
    mask = np.isfinite(r_um) & np.isfinite(z_um) & (r_um >= 0) & (z_um >= 0)
    r_um, z_um = r_um[mask], z_um[mask]
    r_um, z_um = _orient_first_point_at_axis_z(r_um, z_um, r_zero_tol_um=0.5)
    return r_um, z_um

def maybe_load_overlay(cfg: Config) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if not PROFILE_OVERLAY_ENABLE or PROFILE_OVERLAY_PATH is None:
        return None
    p = PROFILE_OVERLAY_PATH
    if not p.exists():
        alt = Path("fitted_profile_radial_for_pinn.npz")
        if alt.exists():
            p = alt
        else:
            print(f"[OVERLAY] Файл профиля не найден: {PROFILE_OVERLAY_PATH}")
            return None
    try:
        return _load_overlay_profile_npz(
            p, cfg,
            prefer_axis=PROFILE_OVERLAY_X_AXIS,
            y_is_depth=PROFILE_Y_AS_DEPTH,
            unit_y_to_m=PROFILE_UNIT_Y_TO_M,
        )
    except Exception as e:
        print(f"[OVERLAY] Не удалось загрузить профиль: {e}")
        return None
# =========================

def plot_T(
    T: np.ndarray,
    cfg: Config,
    tau: float,
    *,
    iso_temp_C: Optional[float],
    out_png: Path,
    vmin=None,
    vmax=None,
    overlay_curve: Optional[Tuple[np.ndarray, np.ndarray]] = None
):
    R_um = float(getattr(cfg, "R_m", 1.0)) * 1e6
    H_um = float(getattr(cfg, "H_m", 1.0)) * 1e6
    Tw_us = float(getattr(cfg, "Twindow_s", 1.0)) * 1e6

    nz, nr = T.shape
    x_um = np.linspace(0, R_um, nr)
    y_um = np.linspace(0, H_um, nz)

    fig, ax = plt.subplots(figsize=(6.0, 4.8), dpi=140)
    im = ax.imshow(T, origin='lower', aspect='auto',
                   extent=[0, R_um, 0, H_um], cmap=CMAP, vmin=vmin, vmax=vmax)
    cb = plt.colorbar(im, ax=ax)
    cb.set_label('T (°C)')

    # Изотермы
    if iso_temp_C is not None:
        draw_iso_contour(ax, x_um, y_um, T, iso_temp_C,
                         color="red", linestyle="-", linewidth=1.25, label=True, snap_to_range=ISO_SNAP_TO_RANGE)
    if ISO_DASH_TEMP_C is not None:
        draw_iso_contour(ax, x_um, y_um, T, ISO_DASH_TEMP_C,
                         color="cyan", linestyle="--", linewidth=1.5, label=True, snap_to_range=ISO_SNAP_TO_RANGE)

    # Наложение профиля бороздки
    if overlay_curve is not None:
        r_um, z_um = overlay_curve
        z_um = z_um + float(PROFILE_Z_OFFSET_UM)
        m = (r_um >= 0) & (r_um <= R_um) & (z_um >= 0) & (z_um <= H_um)
        if np.any(m):
            ax.plot(r_um[m], z_um[m], **PROFILE_STYLE)

    ax.set_xlabel('r (μм)')
    ax.set_ylabel('z (μм)')
    ax.set_title(f"τ={tau:.3f}  (t≈{tau*Tw_us:.2f} μс)  |  Tmin={np.nanmin(T):.2f} °C, Tmax={np.nanmax(T):.2f} °C")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

# ==== авто-оценка глобальных границ колорбара ====
def compute_global_T_limits(
    model: nn.Module,
    cfg: Config,
    taus: List[float],
    RHO: np.ndarray,
    ZETA: np.ndarray,
    device: torch.device,
    deltaT_scale_K: float,
    T0_C: float
) -> Tuple[float, float]:
    t_min = float("+inf");  t_max = float("-inf")
    for tau in taus:
        U = eval_slice_fixed(model, float(tau), RHO, ZETA, device)
        T = T0_C + deltaT_scale_K * U
        t_min = min(t_min, float(np.min(T)))
        t_max = max(t_max, float(np.max(T)))
    if t_min == t_max:
        t_min -= 1.0;  t_max += 1.0
    return t_min, t_max
# ===================================================================

def run_visualization(
    preset_path: Path = PRESET_PATH,
    ckpt_path: Path = CKPT_PATH,
    device: str = DEVICE,
    tau_list: Optional[List[float]] = None,
    nr: int = NR,
    nz: int = NZ,
):
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    cfg, extras = load_cfg_and_extras(preset_path)
    cfg.validate()

    dev = torch.device(device)

    model = build_model_from_cfg(cfg).to(dev)
    src = build_optical_source(Path(preset_path), device=str(dev))

    load_checkpoint_into(model, ckpt_path, dev)
    model.eval()

    deltaT_scale_K = resolve_deltaT_scale_K(cfg, extras, src)
    T0_C = float(getattr(cfg, "T0_C", 25.0))

    RHO, ZETA = np.meshgrid(np.linspace(0, 1, nr, dtype=np.float32),
                            np.linspace(0, 1, nz, dtype=np.float32))

    taus = tau_list if tau_list is not None else TAU_LIST

    # Фиксируем глобальные границы колорбара
    if VMIN is None or VMAX is None:
        auto_vmin, auto_vmax = compute_global_T_limits(
            model=model, cfg=cfg, taus=taus, RHO=RHO, ZETA=ZETA,
            device=dev, deltaT_scale_K=deltaT_scale_K, T0_C=T0_C
        )
        vmin_fixed = VMIN if VMIN is not None else auto_vmin
        vmax_fixed = VMAX if VMAX is not None else auto_vmax
    else:
        vmin_fixed, vmax_fixed = VMIN, VMAX

    # Загружаем профиль для наложения (один раз)
    overlay_curve = maybe_load_overlay(cfg)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    npz = {
        "RHO": RHO.astype(np.float32),
        "ZETA": ZETA.astype(np.float32),
        "r_um": (np.linspace(0, 1, nr, dtype=np.float32) * float(getattr(cfg, "R_m", 1.0)) * 1e6).astype(np.float32),
        "z_um": (np.linspace(0, 1, nz, dtype=np.float32) * float(getattr(cfg, "H_m", 1.0)) * 1e6).astype(np.float32),
        "T0_C": np.float32(T0_C),
        "deltaT_scale_K": np.float32(deltaT_scale_K),
        "vmin_fixed": np.float32(vmin_fixed),
        "vmax_fixed": np.float32(vmax_fixed),
    }

    for tau in taus:
        U = eval_slice_fixed(model, float(tau), RHO, ZETA, dev)
        T = T0_C + deltaT_scale_K * U

        out_png = OUT_DIR / f"field_T_tau{tau:.3f}.png"
        plot_T(T, cfg, float(tau), iso_temp_C=ISO_TEMP_C, out_png=out_png,
               vmin=vmin_fixed, vmax=vmax_fixed, overlay_curve=overlay_curve)

        key = f"tau_{tau:.3f}"
        npz[f"T_{key}"] = T.astype(np.float32)
        npz[f"U_{key}"] = U.astype(np.float32)

    if SAVE_NPZ:
        np.savez_compressed(OUT_DIR / "fields.npz", **npz)
        print(f"Saved NPZ to {OUT_DIR / 'fields.npz'} and PNGs to {OUT_DIR}")
        print(f"[Colorbar] Fixed limits: vmin={vmin_fixed:.3f} °C, vmax={vmax_fixed:.3f} °C")

if __name__ == "__main__":
    run_visualization(PRESET_PATH, CKPT_PATH, DEVICE, TAU_LIST, NR, NZ)
