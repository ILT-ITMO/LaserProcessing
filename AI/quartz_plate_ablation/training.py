# training.py
"""
High-level training utilities for the axisymmetric heat PINN.

This module provides:
  - train_step(): one optimization step over a batch dict (PDE/IC/BC)
  - train(): full training loop with logging, schedulers and early stopping
  - EarlyStopping: patience-based early stopper on a chosen metric (default: val_loss)
  - build_scheduler(): helper to construct a LR scheduler from cfg

Assumptions / contracts with the project:
  * physics.compute_losses(model, batches, cfg, source=...) -> Dict[str, Tensor]
    must include the key 'loss_total' (and optionally sublosses like loss_pde, ...).
  * "batches" is a dict of tensors on the *same device* as the model.
  * "source" is compatible with physics._eval_source (e.g., OpticalSource) or None.
  * Config (from config.py) provides training hyperparameters (optional but supported):
        - lr (float)
        - weight_decay (float)
        - max_epochs (int)
        - grad_clip_norm (float or None)
        - mixed_precision (bool)
        - scheduler (str or None) in {"cosine", "step", "plateau"}
        - scheduler_step_size (int), scheduler_gamma (float)
        - scheduler_Tmax (int) for cosine
        - early_stop_patience (int)
        - early_stop_min_delta (float)
        - ckpt_dir (str or Path)
        - ckpt_every (int) epochs
        - log_every (int) steps

If some fields are missing in cfg, safe defaults are used.
"""
from __future__ import annotations
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

# Local project imports
from config import Config  # type: ignore
from physics import compute_losses  # type: ignore

# Optional: if project already has I/O helpers, we can gracefully import them
try:
    from pinn_io import save_checkpoint as _io_save_ckpt  # type: ignore
except Exception:
    _io_save_ckpt = None


# -----------------------------
# Utilities
# -----------------------------

def _get(cfg: Config, name: str, default):
    return getattr(cfg, name, default)


def to_device_batches(batches: Dict[str, Tuple[torch.Tensor, ...]], device: torch.device) -> Dict[str, Tuple[torch.Tensor, ...]]:
    out: Dict[str, Tuple[torch.Tensor, ...]] = {}
    for k, tup in batches.items():
        out[k] = tuple(t.to(device) for t in tup)  # type: ignore
    return out


# -----------------------------
# Early Stopping
# -----------------------------
class EarlyStopping:
    def __init__(self, patience: int = 50, min_delta: float = 0.0, mode: str = "min"):
        assert mode in {"min", "max"}
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best: Optional[float] = None
        self.count = 0
        self.should_stop = False

    def step(self, value: float) -> bool:
        """Returns True if training should stop."""
        if self.best is None:
            self.best = value
            self.count = 0
            return False
        improved = (value < self.best - self.min_delta) if self.mode == "min" else (value > self.best + self.min_delta)
        if improved:
            self.best = value
            self.count = 0
        else:
            self.count += 1
            if self.count >= self.patience:
                self.should_stop = True
        return self.should_stop


# -----------------------------
# Scheduler factory
# -----------------------------

def build_scheduler(optimizer: torch.optim.Optimizer, cfg: Config, *, steps_per_epoch: Optional[int] = None):
    name = (_get(cfg, "scheduler", None) or None)
    if name is None:
        return None
    name = str(name).lower()
    if name == "cosine":
        T_max = int(_get(cfg, "scheduler_Tmax", _get(cfg, "max_epochs", 200)))
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
    if name == "step":
        step_size = int(_get(cfg, "scheduler_step_size", 50))
        gamma = float(_get(cfg, "scheduler_gamma", 0.5))
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    if name == "plateau":
        patience = int(_get(cfg, "scheduler_patience", 20))
        factor = float(_get(cfg, "scheduler_factor", 0.5))
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=patience, factor=factor, verbose=False)
    # Unknown -> no scheduler
    return None


# -----------------------------
# Train step
# -----------------------------

def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batches: Dict[str, Tuple[torch.Tensor, ...]],
    cfg: Config,
    *,
    source=None,
    scaler: Optional[GradScaler] = None,
) -> Dict[str, float]:
    """One optimization step over a single aggregated batch dict.

    Returns a dict of scalar losses (float) including 'loss_total'.
    """
    model.train()
    optimizer.zero_grad(set_to_none=True)

    mixed = bool(_get(cfg, "mixed_precision", False)) and torch.cuda.is_available()

    if mixed:
        if scaler is None:
            scaler = GradScaler()
        with autocast():
            loss_dict = compute_losses(model, batches, cfg, source=source)
            loss = loss_dict["loss_total"]
        scaler.scale(loss).backward()
        grad_clip = _get(cfg, "grad_clip_norm", None)
        if grad_clip is not None and grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip))
        scaler.step(optimizer)
        scaler.update()
    else:
        loss_dict = compute_losses(model, batches, cfg, source=source)
        loss = loss_dict["loss_total"]
        loss.backward()
        grad_clip = _get(cfg, "grad_clip_norm", None)
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip))
        optimizer.step()

    # Detach to Python floats for logging
    out = {k: float(v.detach().item()) for k, v in loss_dict.items()}
    return out


# -----------------------------
# Training loop
# -----------------------------

def train(
    model: nn.Module,
    cfg: Config,
    *,
    batch_iter: Callable[[int], Iterable[Dict[str, Tuple[torch.Tensor, ...]]]],
    # batch_iter(epoch) -> iterable over per-step batch dicts
    val_provider: Optional[Callable[[], Dict[str, Tuple[torch.Tensor, ...]]]] = None,
    source=None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[object] = None,
    start_epoch: int = 0,
    save_best: bool = True,
) -> Dict[str, float]:
    """
    High-level training loop.

    Parameters
    ----------
    model : nn.Module
        PINN model taking (rho, zeta, tau) -> U.
    cfg : Config
        Configuration object with training hyperparameters.
    batch_iter : Callable
        Function that returns an iterable of training batches for a given epoch.
        Each item is a dict like in physics.compute_losses.
    val_provider : Optional[Callable]
        Function that returns *one* validation batch dict (aggregated) when called.
        If None, validation is skipped and early stopping uses train loss.
    source : optional
        Source compatible with physics._eval_source.
    optimizer : optional
        If None, Adam is created from cfg.lr and cfg.weight_decay.
    scheduler : optional
        If None, will be built from cfg via build_scheduler().
    start_epoch : int
        Epoch to start with (useful for resumed training).
    save_best : bool
        If True, the best checkpoint is stored at ckpt_dir/best.pt.

    Returns
    -------
    Dict[str, float]: summary with final train/val losses and best metrics.
    """
    device = next(model.parameters()).device

    if optimizer is None:
        lr = float(_get(cfg, "lr", 1e-3))
        wd = float(_get(cfg, "weight_decay", 0.0))
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    # Scheduler
    if scheduler is None:
        scheduler = build_scheduler(optimizer, cfg)

    # Early stopping
    patience = int(_get(cfg, "early_stop_patience", 50))
    min_delta = float(_get(cfg, "early_stop_min_delta", 0.0))
    stopper = EarlyStopping(patience=patience, min_delta=min_delta, mode="min")

    # Grad scaler for AMP
    scaler = GradScaler(enabled=bool(_get(cfg, "mixed_precision", False)) and torch.cuda.is_available())

    # Logging / checkpoints
    ckpt_dir = Path(str(_get(cfg, "ckpt_dir", "checkpoints")))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_every = int(_get(cfg, "ckpt_every", 10))
    log_every = max(1, int(_get(cfg, "log_every", 50)))

    max_epochs = int(_get(cfg, "max_epochs", 500))
    global_step = 0
    best_val = float("inf")

    for epoch in range(start_epoch, max_epochs):
        # ---- Training epoch ----
        epoch_loss_sum = 0.0
        steps = 0
        for step_batches in batch_iter(epoch):
            # ensure device
            step_batches = to_device_batches(step_batches, device)

            loss_scalars = train_step(
                model, optimizer, step_batches, cfg, source=source, scaler=scaler
            )
            epoch_loss_sum += loss_scalars["loss_total"]
            steps += 1
            global_step += 1

            if global_step % log_every == 0:
                print(f"[epoch {epoch} step {global_step}] loss_total={loss_scalars['loss_total']:.4e} "
                      f"pde={loss_scalars.get('loss_pde', float('nan')):.4e} "
                      f"ic={loss_scalars.get('loss_ic', float('nan')):.4e} "
                      f"bc={loss_scalars.get('loss_bc', float('nan')):.4e}")

            # Per-step scheduler (rare; most use per-epoch). If you need, uncomment:
            # if isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
            #     scheduler.step()

        train_epoch_loss = epoch_loss_sum / max(1, steps)

        # ---- Validation ----
        if val_provider is not None:
            model.eval()
            # ВАЖНО: нужны производные для PDE → не используем no_grad.
            with torch.set_grad_enabled(True):  # или просто убери контекст целиком
                val_batches = to_device_batches(val_provider(), device)
                val_losses = compute_losses(model, val_batches, cfg, source=source)
                val_loss = float(val_losses["loss_total"].detach().item())
        else:
            val_loss = train_epoch_loss

        # ---- Scheduler step ----
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # ---- Early stopping ----
        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            if save_best:
                _save_checkpoint(ckpt_dir / "best.pt", model, optimizer, epoch, best_val)
        if stopper.step(val_loss):
            print(f"Early stopping at epoch {epoch} (best val={best_val:.4e})")
            break

        # ---- Epoch checkpoint ----
        if ckpt_every > 0 and ((epoch + 1) % ckpt_every == 0):
            _save_checkpoint(ckpt_dir / f"epoch_{epoch+1}.pt", model, optimizer, epoch, best_val)

        # ---- Epoch log ----
        print(f"Epoch {epoch} | train_loss={train_epoch_loss:.4e} | val_loss={val_loss:.4e} | lr={optimizer.param_groups[0]['lr']:.3e}")

    # Final save (last.pt)
    _save_checkpoint(ckpt_dir / "last.pt", model, optimizer, epoch, best_val)

    return {
        "best_val": best_val,
        "last_epoch": epoch,
        "global_step": global_step,
    }


# -----------------------------
# Checkpoint helpers
# -----------------------------

def _save_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, best_val: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_val": best_val,
        "torch_version": torch.__version__,
    }
    if _io_save_ckpt is not None:
        try:
            _io_save_ckpt(path, state)  # project-specific saver, if present
            return
        except Exception:
            pass
    torch.save(state, path)

