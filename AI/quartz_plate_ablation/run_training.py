# run_training.py
"""
Launcher to train the axisymmetric heat PINN using a JSON preset *and* the
same sampling rules you use elsewhere in the project.

Minimal version (no argparse):
- Set PRESET_PATH / DEVICE / SEED in the config block below or call
  run_training(preset_path, device, seed) from your own script/IDE.
- Keeps cylindrical sampling and time samplers aligned with your preset via
  source.load_cfg_and_extras and (optionally) source.sample_collocation_from_cfg.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Iterable, Tuple, Callable, Optional, Any

import torch
import torch.nn as nn

from config import Config
from training import train
from physics import compute_losses
from source import build_optical_source, load_cfg_and_extras

# ---------- User-config block (edit here as needed) ----------
PRESET_PATH = Path("presets_params/pinn_params_P3p3W_V40mms_20250911_164958.json")   # <- set your preset file
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

# ---------- Utilities ----------

def _get(obj: Any, name: str, default):
    return getattr(obj, name, default) if hasattr(obj, name) else (obj.get(name, default) if isinstance(obj, dict) else default)


def seed_everything(seed: int = 42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------- Model factory ----------

def build_model_from_cfg(cfg: Config) -> nn.Module:
    import inspect
    import models  # project file

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
    raise RuntimeError("Could not construct a model from models.py. Provide build_model(cfg) or set cfg.model_name.")


# ---------- Tau samplers ----------

def sample_tau_uniform(N: int, *, device: torch.device, requires_grad: bool = True) -> torch.Tensor:
    return torch.rand(N, 1, device=device, requires_grad=requires_grad)


def sample_tau_impulse(
    N: int,
    *,
    device: torch.device,
    t0_tau: float = 0.5,
    sigma_tau: float = 0.15,
    mix: float = 0.6,
    requires_grad: bool = True,
) -> torch.Tensor:
    tau_uni = torch.rand(N, 1, device=device)
    normal = torch.distributions.Normal(
        torch.tensor([t0_tau], device=device), torch.tensor([sigma_tau], device=device)
    )
    tau_imp = normal.sample((N,))
    tau = mix * tau_imp + (1.0 - mix) * tau_uni
    tau = tau.clamp_(0.0, 1.0)
    tau.requires_grad_(requires_grad)
    return tau


def build_tau_sampler(extras: Dict[str, Any], device: torch.device):
    samp = extras.get("sampling", {})
    tau_cfg = samp.get("tau", {})

    if "tau_grid" in tau_cfg and isinstance(tau_cfg["tau_grid"], list) and len(tau_cfg["tau_grid"]) > 0:
        tau_grid = torch.tensor(tau_cfg["tau_grid"], dtype=torch.float32, device=device).clamp(0, 1)
        def _grid(N: int) -> torch.Tensor:
            reps = int((N + len(tau_grid) - 1) // len(tau_grid))
            vals = tau_grid.repeat(reps)[:N].view(N, 1).clone().requires_grad_(True)
            return vals
        return _grid

    mode = str(tau_cfg.get("mode", "impulse_mix")).lower()
    if mode == "uniform":
        return lambda N: sample_tau_uniform(N, device=device)
    t0 = float(tau_cfg.get("t0_tau", 0.5))
    sig = float(tau_cfg.get("sigma_tau", 0.15))
    mix = float(tau_cfg.get("mix", 0.6))
    return lambda N: sample_tau_impulse(N, device=device, t0_tau=t0, sigma_tau=sig, mix=mix)


# ---------- Spatial samplers ----------
Batch = Dict[str, Tuple[torch.Tensor, ...]]


def _fallback_collocation(cfg: Config, N: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    rho = (torch.rand(N, 1, device=device) * (1.0 - 2e-4) + 1e-4).requires_grad_(True)
    zeta = torch.rand(N, 1, device=device).requires_grad_(True)
    return rho, zeta


def _collocation_from_source(cfg: Config, N: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    try:
        from source import sample_collocation_from_cfg  # type: ignore
        rho, zeta, _tau = sample_collocation_from_cfg(cfg, N=N, device=device)
        rho = rho.detach().clone().requires_grad_(True).to(device)
        zeta = zeta.detach().clone().requires_grad_(True).to(device)
        return rho, zeta
    except Exception:
        return _fallback_collocation(cfg, N, device)


def _make_batch_once(cfg: Config, extras: Dict[str, Any], device: torch.device, tau_sampler) -> Batch:
    N_coll = int(extras.get("N_coll", _get(cfg, "N_coll", 8192)))
    N_ic = int(extras.get("N_ic", _get(cfg, "N_ic", 1024)))
    N_bc_axis = int(extras.get("N_bc_axis", _get(cfg, "N_bc_axis", 1024)))
    N_bc_wall = int(extras.get("N_bc_wall", _get(cfg, "N_bc_wall", 1024)))
    N_bc_z0 = int(extras.get("N_bc_z0", _get(cfg, "N_bc_z0", 1024)))
    N_bc_z1 = int(extras.get("N_bc_z1", _get(cfg, "N_bc_z1", 1024)))

    rho_c, zeta_c = _collocation_from_source(cfg, N_coll, device)
    tau_c = tau_sampler(N_coll)

    rho_ic = torch.rand(N_ic, 1, device=device).requires_grad_(True)
    zeta_ic = torch.rand(N_ic, 1, device=device).requires_grad_(True)
    u_ic_target = torch.zeros(N_ic, 1, device=device)

    zeta_axis = torch.rand(N_bc_axis, 1, device=device).requires_grad_(True)
    tau_axis = tau_sampler(N_bc_axis)

    zeta_wall = torch.rand(N_bc_wall, 1, device=device).requires_grad_(True)
    tau_wall = tau_sampler(N_bc_wall)

    rho_z0 = torch.rand(N_bc_z0, 1, device=device).requires_grad_(True)
    tau_z0 = tau_sampler(N_bc_z0)

    rho_z1 = torch.rand(N_bc_z1, 1, device=device).requires_grad_(True)
    tau_z1 = tau_sampler(N_bc_z1)

    return {
        "pde": (rho_c, zeta_c, tau_c),
        "ic": (rho_ic, zeta_ic, u_ic_target),
        "axis": (zeta_axis, tau_axis),
        "wall": (zeta_wall, tau_wall),
        "z0": (rho_z0, tau_z0),
        "z1": (rho_z1, tau_z1),
    }


def make_batch_iter(cfg: Config, extras: Dict[str, Any], device: torch.device):
    steps_per_epoch = int(extras.get("steps_per_epoch", _get(cfg, "steps_per_epoch", 100)))
    tau_sampler = build_tau_sampler(extras, device)

    try:
        import pinn_io
        if hasattr(pinn_io, "batch_iter"):
            return lambda epoch: pinn_io.batch_iter(epoch, cfg=cfg, device=device)  # type: ignore
        if hasattr(pinn_io, "make_batch_iter"):
            return pinn_io.make_batch_iter(cfg=cfg, device=device)  # type: ignore
    except Exception:
        pass

    def _iter(_epoch: int):
        for _ in range(steps_per_epoch):
            yield _make_batch_once(cfg, extras, device, tau_sampler)
    return _iter


def make_val_provider(cfg: Config, extras: Dict[str, Any], device: torch.device):
    try:
        import pinn_io
        if hasattr(pinn_io, "val_provider"):
            return lambda: pinn_io.val_provider(cfg=cfg, device=device)  # type: ignore
    except Exception:
        pass
    tau_sampler = build_tau_sampler(extras, device)
    fixed = _make_batch_once(cfg, extras, device, tau_sampler)
    return lambda: fixed


# ---------- Public runner (call this from IDE or other scripts) ----------

def run_training(preset_path: Path = PRESET_PATH, device: str = DEVICE, seed: int = SEED):
    seed_everything(seed)

    cfg, extras = load_cfg_and_extras(preset_path)
    cfg.validate()

    dev = torch.device(device)
    model = build_model_from_cfg(cfg).to(dev)
    src = build_optical_source(Path(preset_path), device=str(dev))

    batch_iter = make_batch_iter(cfg, extras, dev)
    val_provider = make_val_provider(cfg, extras, dev)

    summary = train(
        model=model,
        cfg=cfg,
        batch_iter=batch_iter,
        val_provider=val_provider,
        source=src,
    )
    print("Training finished:", summary)


# ---------- Script entry ----------
if __name__ == "__main__":
    run_training(PRESET_PATH, DEVICE, SEED)
