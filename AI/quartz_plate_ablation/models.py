# pinn_heat/models.py
"""
Модели для PINN.
Добавлено:
- активация 'sine' (SIREN) с профильной инициализацией;
- адаптивный масштаб активации (ScaledActivation) для любого act, напр. tanh/sine.
Ссылки: Sitzmann et al., NeurIPS'20 (SIREN); Jagtap et al., Proc. Royal Soc. A'20 (adaptive).
"""

from typing import Iterable, Tuple, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------- #
# АКТИВАЦИИ И ИНИЦИАЛИЗАЦИЯ
# ------------------------------- #

class Sine(nn.Module):
    """Периодическая активация SIREN: y = sin(x)."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)


class ScaledActivation(nn.Module):
    """Обучаемый масштаб α для любой активации: act(α·x)."""
    def __init__(self, act: nn.Module, alpha_init: float = 1.0):
        super().__init__()
        self.act = act
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.alpha * x)


def get_activation(name: str) -> nn.Module:
    """'tanh' | 'relu' | 'gelu' | 'silu' | 'sine'."""
    name = (name or "tanh").lower()
    if name == "tanh":
        return nn.Tanh()
    if name == "relu":
        return nn.ReLU(inplace=False)
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU(inplace=False)
    if name == "sine":
        return Sine()
    raise ValueError(f"Unknown activation: {name}")


def init_linear(layer: nn.Linear, nonlinearity: str = "tanh", is_first_sine: bool = False) -> None:
    """
    Инициализация слоёв.
    - Для 'sine' (SIREN): Uniform(-1/fan_in, 1/fan_in) для всех слоёв,
      а первый слой обычно масштабируют сильнее (см. SIREN).
    - Для 'tanh': Xavier uniform.
    - Для ReLU/SiLU/GELU: Kaiming uniform.
    """
    nl = nonlinearity.lower()
    fan_in = layer.weight.shape[1]

    if nl == "sine":
        # Базовая инициализация SIREN (см. Sitzmann et al. 2020)
        # Первый слой обычно берут U(-1/fan_in, 1/fan_in); последующие иногда домножают на 30^-1 и т.п.
        bound = 1.0 / fan_in
        nn.init.uniform_(layer.weight, -bound, bound)
        nn.init.zeros_(layer.bias)
        return

    if nl == "tanh":
        nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain("tanh"))
        nn.init.zeros_(layer.bias)
        return

    if nl in {"relu", "silu", "gelu"}:
        nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
        nn.init.zeros_(layer.bias)
        return

    # по умолчанию — Xavier
    nn.init.xavier_uniform_(layer.weight)
    nn.init.zeros_(layer.bias)


# ------------------------------- #
# МОДЕЛЬ
# ------------------------------- #

class MLP(nn.Module):
    """
    MLP: u(rho, zeta, tau) -> U (безразмерная температура).
    Опции:
      - activation: 'tanh' (по умолчанию), 'sine', 'relu'/'silu'/'gelu'
      - adaptive_scale: обучаемый α в каждом скрытом слое: act(α·x)
      - nonneg: при True применяет softplus на выходе
    """

    def __init__(
        self,
        in_dim: int = 3,
        widths: Iterable[int] = (64, 64, 64, 64),
        out_dim: int = 1,
        activation: str = "tanh",
        nonneg: bool = True,
        adaptive_scale: bool = False,
        alpha_init: float = 1.0,
        last_linear_bias_init: Optional[float] = 0.0,
    ):
        super().__init__()
        self.nonneg = nonneg
        self.activation_name = activation
        self.adaptive_scale = adaptive_scale
        self.alpha_init = alpha_init

        layers = []
        last = in_dim
        act_base = get_activation(activation)

        for li, w in enumerate(widths):
            lin = nn.Linear(last, int(w))
            init_linear(lin, nonlinearity=activation, is_first_sine=(activation.lower() == "sine" and li == 0))

            # при необходимости обернём активацию в обучаемый масштаб
            act: nn.Module = act_base if not adaptive_scale else ScaledActivation(get_activation(activation), alpha_init)

            layers += [lin, act]
            last = int(w)

        self.backbone = nn.Sequential(*layers)

        self.head = nn.Linear(last, out_dim)
        # Последний слой — «мягкая» Xavier
        nn.init.xavier_uniform_(self.head.weight)
        if last_linear_bias_init is not None:
            nn.init.constant_(self.head.bias, float(last_linear_bias_init))
        else:
            nn.init.zeros_(self.head.bias)

    def forward(self, rho: torch.Tensor, zeta: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        x = torch.cat([rho, zeta, tau], dim=1)
        y = self.backbone(x)
        y = self.head(y)
        return F.softplus(y) if self.nonneg else y

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ------------------------------- #
# ФАБРИКА
# ------------------------------- #

def build_model(cfg) -> nn.Module:
    """
    Читает из cfg:
      - widths: Tuple[int,...]
      - activation: str ('tanh' | 'sine' | 'relu' | 'silu' | 'gelu')
      - adaptive_activation (bool, опц.): вкл/выкл обучаемый масштаб α
      - alpha_init (float, опц.): начальное значение α
      - nonneg (bool, опц.)
    """
    widths: Tuple[int, ...] = tuple(getattr(cfg, "widths", (64, 64, 64, 64)))
    activation: str = getattr(cfg, "activation", "tanh")
    nonneg: bool = bool(getattr(cfg, "nonneg", True))
    adaptive: bool = bool(getattr(cfg, "adaptive_activation", False))
    alpha_init: float = float(getattr(cfg, "alpha_init", 1.0))

    return MLP(
        in_dim=3,
        widths=widths,
        out_dim=1,
        activation=activation,
        nonneg=nonneg,
        adaptive_scale=adaptive,
        alpha_init=alpha_init,
        last_linear_bias_init=0.0,
    )
