import pytest
import torch
from pinn import PINN, compute_pinn_loss

def test_pinn_initialization():
    layers = [4, 20, 20, 1]
    model = PINN(layers)
    assert len(model.network) > 0

def test_pinn_forward_pass():
    model = PINN([4, 10, 1])
    # Batch size of 5
    x = torch.randn(5)
    y = torch.randn(5)
    z = torch.randn(5)
    t = torch.randn(5)
    
    output = model(x, y, z, t)
    assert output.shape == (5, 1)

def test_compute_loss_runs():
    # Smoke test for loss computation
    model = PINN([4, 10, 1])
    x = torch.zeros(5)
    y = torch.zeros(5)
    z = torch.zeros(5)
    t = torch.zeros(5)
    
    # Should not raise error
    loss = compute_pinn_loss(model, x, y, z, t, diff_coef=1.0, laser_mode="continuous")
    assert isinstance(loss, torch.Tensor)
    assert not torch.isnan(loss)
