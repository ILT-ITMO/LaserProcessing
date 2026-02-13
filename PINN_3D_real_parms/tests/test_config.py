import pytest
import os
import json
from config import LaserConfig

def test_config_initialization(test_config):
    assert test_config.config is not None
    assert "laser" in test_config.config
    assert "material" in test_config.config

def test_derived_parameters(test_config):
    # Check if calculation works
    test_config.calculate_derived_parameters()
    assert hasattr(test_config, "CHARACTERISTIC_LENGTH")
    assert hasattr(test_config, "THERMAL_DIFFUSIVITY")
    assert test_config.LASER_MODE in ["pulsed", "continuous"]

def test_save_load_config(test_config, tmp_path):
    # Test saving and loading
    filepath = tmp_path / "test_config.json"
    
    # Modify a value
    test_config.config["laser"]["avg_power"] = 999.0
    test_config.save_to_json(str(filepath))
    
    # Load back
    new_config = LaserConfig()
    new_config.load_from_json(str(filepath))
    
    assert new_config.config["laser"]["avg_power"] == 999.0
