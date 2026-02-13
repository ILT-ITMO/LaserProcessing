import pytest
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import LaserConfig

@pytest.fixture
def test_config():
    """Returns a LaserConfig instance with default settings"""
    return LaserConfig()

@pytest.fixture
def small_pinn_config():
    """Returns a config optimized for quick testing"""
    cfg = LaserConfig()
    cfg.config["training"]["num_epochs"] = 2
    cfg.config["pinn"]["collocation_points"] = {"x": 5, "y": 5, "z": 5, "t": 5}
    cfg.calculate_derived_parameters()
    return cfg
