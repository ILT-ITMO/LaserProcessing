# plot_iso_profiles.py
# Читает профили изотермы из viz/iso_profiles/iso_tau*.csv и строит график + накладывает профиль бороздки.
from __future__ import annotations
from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ====== РЕДАКТИРУЕМЫЕ ПАРАМЕТРЫ ======
INPUT_DIR = Path("viz/iso_profiles")   # Папка с iso_tau*.csv
FILE_GLOB = "iso_tau*.csv"             # Паттерн файлов
MODE = "overlay"                        # "overlay" или "per_tau"
ASPECT_EQUAL = False                    # True → соотношение осей 1:1
SHOW_COLORBAR = True                    # Цветовая шкала по τ
SHOW_SURFACE_LINE = True                # Линия z=0
SHOW_LEGEND = True                      # Легенда (для профиля)
SAVE_FIG = True
OUT_PATH = Path("viz/iso_profiles_plot.png")
DPI = 200
LINEWIDTH = 1.4
SELECT_TAU: Optional[List[float]] = None   # например [0.10, 0.25]

# ==== OVERLAY: профиль бороздки ====
OVERLAY_ENABLE = True
OVERLAY_PATH = Path("fitted_profile_radial_for_pinn.npz")
OVERLAY_X_SOURCE = "auto"          # понимает r_um/x_um/r_m/x_m/rho
OVERLAY_Y_AS_DEPTH = False         # y_fit — высота → считаем глубину baseline - y
OVERLAY_UNIT_Y_TO_UM = 1.0         # если y в μм → 1.0
OVERLAY_Z_OFFSET_UM = 0.0          # вертикальный сдвиг профиля (μм)
OVERLAY_STYLE = dict(color="tab:red", linewidth=2.0, alpha=0.95, label="эксп. профиль")
# ==================================

def parse_meta(path: Path) -> Dict[str, float]:
    meta = {"iso_temp_C": np.nan, "tau": np.nan, "t_us": np.nan}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.startswith("#"):
                break
            m = re.search(r"iso_temp_C\s*=\s*([+-]?\d+(?:\.\d+)?)", line)
            if m: meta["iso_temp_C"] = float(m.group(1))
            m = re.search(r"tau\s*=\s*([+-]?\d+(?:\.\d+)?)", line)
            if m: meta["tau"] = float(m.group(1))
            m = re.search(r"t_us\s*=\s*([+-]?\d+(?:\.\d+)?)", line)
            if m: meta["t_us"] = float(m.group(1))
    return meta

def load_profile_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path, comment="#", header=None,
        names=["seg_id", "x_um", "y_um"],
        dtype={"seg_id": int, "x_um": float, "y_um": float},
    )

def collect_files() -> List[Tuple[Path, Dict[str, float], pd.DataFrame]]:
    files = sorted(INPUT_DIR.glob(FILE_GLOB))
    items: List[Tuple[Path, Dict[str, float], pd.DataFrame]] = []
    for p in files:
        try:
            meta = parse_meta(p)
            if SELECT_TAU is not None and not any(abs(meta["tau"] - t) < 1e-9 for t in SELECT_TAU):
                continue
            df = load_profile_csv(p)
            if not df.empty:
                items.append((p, meta, df))
        except Exception as e:
            print(f"[WARN] Пропускаю {p.name}: {e}")
    items.sort(key=lambda x: (np.inf if np.isnan(x[1]["tau"]) else x[1]["tau"]))
    return items

# ==== OVERLAY helpers ====
def _orient_first_point_at_axis_z(r_um: np.ndarray, z_um: np.ndarray, *, r_zero_tol_um: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    if r_um.size == 0:
        return r_um, z_um
    idx = int(np.nanargmin(r_um))
    if idx != 0:
        r_um = np.concatenate([r_um[idx:], r_um[:idx]])
        z_um = np.concatenate([z_um[idx:], z_um[:idx]])
    if np.nanmedian(np.diff(r_um)) < 0:
        r_um = r_um[::-1];  z_um = z_um[::-1]
    if r_um[0] <= max(0.0, float(r_zero_tol_um)):
        r_um[0] = 0.0
    return r_um, z_um

def _load_overlay_profile_npz(
    path: Path,
    *,
    x_source: str = "auto",
    y_as_depth: bool = False,
    unit_y_to_um: float = 1.0,
    fallback_R_um: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Загружает профиль из NPZ и возвращает (r_um, z_um) с правильными осями:
      X — радиус r (μм), Y — высота → глубина z = baseline - y (положительная вниз).
    """
    if not path.exists():
        raise FileNotFoundError(f"Файл профиля не найден: {path}")
    with np.load(path, allow_pickle=True) as data:
        # --- X
        key = None
        if x_source != "auto" and x_source in data:
            key = x_source
        else:
            for cand in ("r_um", "x_um", "r_m", "x_m", "rho"):
                if cand in data:
                    key = cand; break
        if key is None:
            raise KeyError("В NPZ нет ни 'r_um', 'x_um', 'r_m', 'x_m', ни 'rho'.")

        if key in ("r_um", "x_um"):
            r_um = np.asarray(data[key], float)
        elif key in ("r_m", "x_m"):
            r_um = np.asarray(data[key], float) * 1e6
        elif key == "rho":
            if fallback_R_um is None or not np.isfinite(fallback_R_um):
                raise ValueError("Для 'rho' требуется fallback_R_um (макс. r по изотермам).")
            r_um = np.asarray(data["rho"], float) * float(fallback_R_um)

        # --- Y
        if "y_fit" in data:
            y = np.asarray(data["y_fit"], float)
        elif "y" in data:
            y = np.asarray(data["y"], float)
        else:
            raise KeyError("В NPZ нет ни 'y_fit', ни 'y'.")

        # высота/глубина → глубина в μм (положительная вниз)
        if y_as_depth:
            depth_um = y * float(unit_y_to_um)
        else:
            baseline = None
            try:
                if "params" in data:
                    params = data["params"].item()
                    baseline = float(params.get("baseline", np.nan))
            except Exception:
                baseline = None
            if baseline is None or not np.isfinite(baseline):
                baseline = float(np.nanmax(y))
            depth_um = (baseline - y) * float(unit_y_to_um)

    # Сделаем глубину положительной
    if np.nanmedian(depth_um) < 0:
        depth_um = -depth_um

    # фильтрация и ориентация (первая точка — на оси z)
    m = np.isfinite(r_um) & np.isfinite(depth_um) & (r_um >= 0)
    r_um, depth_um = r_um[m], np.maximum(depth_um[m], 0.0)
    r_um, depth_um = _orient_first_point_at_axis_z(r_um, depth_um, r_zero_tol_um=0.5)
    return r_um, depth_um
# =========================

def plot_overlay(items: List[Tuple[Path, Dict[str, float], pd.DataFrame]]) -> plt.Figure:
    if not items:
        raise RuntimeError("Нет подходящих файлов профилей изотермы.")

    taus = np.array([it[1]["tau"] for it in items if not np.isnan(it[1]["tau"])], dtype=float)
    have_tau = taus.size > 0
    cm = plt.get_cmap("viridis")
    norm = plt.Normalize(vmin=float(np.min(taus)) if have_tau else 0.0,
                         vmax=float(np.max(taus)) if have_tau else 1.0)

    fig, ax = plt.subplots(figsize=(7, 5), dpi=DPI)

    # Рисуем изотермы
    xmax_all = 0.0
    for _, meta, df in items:
        color = cm(norm(meta["tau"])) if have_tau else None
        for _, grp in df.groupby("seg_id"):
            ax.plot(grp["x_um"].values, grp["y_um"].values,
                    linewidth=LINEWIDTH, color=color, alpha=0.95)
            if grp["x_um"].values.size:
                xmax_all = max(xmax_all, float(np.nanmax(grp["x_um"].values)))

    if SHOW_SURFACE_LINE:
        ax.axhline(0.0, linestyle=":", linewidth=1.0, color="k", alpha=0.5)

    # Наложение профиля бороздки
    if OVERLAY_ENABLE:
        try:
            r_um, z_um = _load_overlay_profile_npz(
                OVERLAY_PATH,
                x_source=OVERLAY_X_SOURCE,
                y_as_depth=OVERLAY_Y_AS_DEPTH,
                unit_y_to_um=OVERLAY_UNIT_Y_TO_UM,
                fallback_R_um=xmax_all if np.isfinite(xmax_all) and xmax_all > 0 else None,
            )
            z_um = z_um + float(OVERLAY_Z_OFFSET_UM)
            m = (r_um >= 0) & (z_um >= 0)
            if np.any(m):
                ax.plot(r_um[m], z_um[m], **OVERLAY_STYLE)
            else:
                print("[OVERLAY] После фильтрации точек для наложения не осталось.")
        except Exception as e:
            print(f"[OVERLAY] Не удалось наложить профиль: {e}")

    # Оформление
    iso_vals = [it[1]["iso_temp_C"] for it in items if not np.isnan(it[1]["iso_temp_C"])]
    title = "Профили изотермы"
    if iso_vals and (np.nanmax(iso_vals) - np.nanmin(iso_vals) < 1e-9):
        title += f"  T≈{iso_vals[0]:.2f} °C"
    ax.set_title(title)

    ax.set_xlabel("r (μм)")
    ax.set_ylabel("z (μм)")
    ax.grid(True, alpha=0.25)
    if ASPECT_EQUAL:
        ax.set_aspect("equal", adjustable="box")

    if SHOW_COLORBAR and have_tau:
        sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax)
        cb.set_label("τ")

    if SHOW_LEGEND:
        handles, labels = ax.get_legend_handles_labels()
        if handles and labels:
            ax.legend(loc="best")

    fig.tight_layout()
    return fig

def plot_per_tau(items: List[Tuple[Path, Dict[str, float], pd.DataFrame]]) -> List[plt.Figure]:
    # Предзагрузка профиля (тот же для всех τ)
    overlay_curve: Optional[Tuple[np.ndarray, np.ndarray]] = None
    xmax_all = 0.0
    for _, _, df in items:
        if df["x_um"].values.size:
            xmax_all = max(xmax_all, float(np.nanmax(df["x_um"].values)))
    if OVERLAY_ENABLE:
        try:
            overlay_curve = _load_overlay_profile_npz(
                OVERLAY_PATH,
                x_source=OVERLAY_X_SOURCE,
                y_as_depth=OVERLAY_Y_AS_DEPTH,
                unit_y_to_um=OVERLAY_UNIT_Y_TO_UM,
                fallback_R_um=xmax_all if np.isfinite(xmax_all) and xmax_all > 0 else None,
            )
        except Exception as e:
            print(f"[OVERLAY] Не удалось загрузить профиль для per_tau: {e}")
            overlay_curve = None

    figs: List[plt.Figure] = []
    for p, meta, df in items:
        fig, ax = plt.subplots(figsize=(6, 4.5), dpi=DPI)
        for _, grp in df.groupby("seg_id"):
            ax.plot(grp["x_um"].values, grp["y_um"].values, linewidth=LINEWIDTH)

        if SHOW_SURFACE_LINE:
            ax.axhline(0.0, linestyle=":", linewidth=1.0, color="k", alpha=0.5)

        if overlay_curve is not None:
            r_um, z_um = overlay_curve
            z_um2 = z_um + float(OVERLAY_Z_OFFSET_UM)
            m = (r_um >= 0) & (z_um2 >= 0)
            if np.any(m):
                ax.plot(r_um[m], z_um2[m], **OVERLAY_STYLE)

        parts = []
        if not np.isnan(meta["tau"]):   parts.append(f"τ={meta['tau']:.3f}")
        if not np.isnan(meta["t_us"]):  parts.append(f"t≈{meta['t_us']:.2f} μс")
        if not np.isnan(meta["iso_temp_C"]): parts.append(f"T≈{meta['iso_temp_C']:.2f} °C")
        ax.set_title(" | ".join(parts) if parts else p.name)

        ax.set_xlabel("r (μм)")
        ax.set_ylabel("z (μм)")
        ax.grid(True, alpha=0.25)
        if ASPECT_EQUAL:
            ax.set_aspect("equal", adjustable="box")

        if SHOW_LEGEND:
            handles, labels = ax.get_legend_handles_labels()
            if handles and labels:
                ax.legend(loc="best")

        fig.tight_layout()
        figs.append(fig)
        if SAVE_FIG:
            out = OUT_PATH.with_name(f"{OUT_PATH.stem}_{meta['tau']:.3f}.png") if not np.isnan(meta["tau"]) else OUT_PATH.with_name(f"{OUT_PATH.stem}_{p.stem}.png")
            fig.savefig(out, dpi=DPI)
            print(f"[SAVE] {out}")
    return figs

def main():
    items = collect_files()
    if not items:
        print(f"[INFO] Файлы не найдены в {INPUT_DIR / FILE_GLOB}")
        return

    if MODE == "per_tau":
        figs = plot_per_tau(items)
        if not SAVE_FIG:
            plt.show()
        else:
            for fig in figs:
                plt.close(fig)
    else:
        fig = plot_overlay(items)
        if SAVE_FIG:
            OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(OUT_PATH, dpi=DPI)
            print(f"[SAVE] {OUT_PATH}")
            plt.close(fig)
        else:
            plt.show()

if __name__ == "__main__":
    main()
