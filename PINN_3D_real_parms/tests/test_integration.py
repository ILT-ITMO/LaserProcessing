import pytest
import os
from pinn import train_pinn, PINN
import torch

def test_training_loop_callback(small_pinn_config):
    # Verify training loop calls the callback
    model = PINN([4, 10, 1])
    diff_coef = 1.0
    
    callback_called = False
    
    def progress_cb(epoch, loss):
        nonlocal callback_called
        callback_called = True
        assert isinstance(epoch, int)
        assert isinstance(loss, float)
    
    history = train_pinn(
        model, 
        diff_coef, 
        num_epochs=2, 
        lr=1e-3, 
        device='cpu', 
        laser_mode="continuous",
        progress_callback=progress_cb
    )
    
    assert len(history) == 2
    assert callback_called

def test_full_integration(tmp_path):
    # Test file artifact generation
    # Since main.py writes to 'results/' in CWD, we might need to mock or change CWD
    # For now, just test that train_pinn completes without error
    model = PINN([4, 10, 1])
    history = train_pinn(model, 1.0, num_epochs=1, device='cpu', laser_mode="continuous")
    assert len(history) == 1
