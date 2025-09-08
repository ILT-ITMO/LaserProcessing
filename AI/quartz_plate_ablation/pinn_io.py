# ===============================
# pinn_heat/pinn_io.py (simplified)
# ===============================
"""
Простой модуль ввода-вывода:
- JSON с параметрами задачи (pinn_params.json)
- NPZ с полем температуры (U_series.npz)
Минимум зависимостей, без атомарной записи и служебных утилит.
"""

from pathlib import Path
import json
from typing import Any, Dict, Tuple, Optional

import numpy as np

from config import Config


# ------------------------------- #
# ПАРАМЕТРЫ ЗАДАЧИ (JSON)
# ------------------------------- #

def save_params_json(path: Path, cfg: Config, *, extra: Optional[Dict[str, Any]] = None, pretty: bool = True) -> None:
    """Сохранить dataclass Config в JSON. Добавляет DELTA_T_SCALE_K, если доступен."""
    path = Path(path)
    data: Dict[str, Any] = cfg.__dict__.copy()

    # по возможности положим рассчитанный масштаб ΔT
    if hasattr(cfg, "deltaT_scale_K"):
        try:
            data["DELTA_T_SCALE_K"] = float(cfg.deltaT_scale_K())
        except Exception:
            pass

    if extra:
        data.update(extra)

    text = json.dumps(data, ensure_ascii=False, indent=2 if pretty else None)
    path.write_text(text, encoding="utf-8")


def load_params_json(path: Path) -> Config:
    """Загрузить JSON и сконструировать Config. Лишние поля игнорируются."""
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    cfg_kwargs = {k: v for k, v in data.items() if k in Config.__annotations__}
    cfg = Config(**cfg_kwargs)
    if hasattr(cfg, "validate"):
        cfg.validate()
    return cfg


# ------------------------------- #
# РЕЗУЛЬТАТЫ РАСЧЁТА (NPZ)
# ------------------------------- #

def save_u_series_npz(path: Path, U_3d: np.ndarray, tau: np.ndarray, t_sec: Optional[np.ndarray]) -> None:
    """Сохранить поле U и временные вектора в сжатый NPZ.
    U_3d: (N_tau, N_r, N_z), tau: (N_tau,), t_sec: (N_tau,) или None
    """
    path = Path(path)
    np.savez_compressed(path, U=U_3d, tau=tau, t_sec=t_sec)


def load_u_series_npz(path: Path) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Загрузить (U, tau, t_sec) из NPZ."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        U = data["U"]
        tau = data["tau"]
        t_sec = data["t_sec"] if "t_sec" in data and data["t_sec"] is not None else None
    return U, tau, t_sec

