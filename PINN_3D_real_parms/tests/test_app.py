from fastapi.testclient import TestClient
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import app

client = TestClient(app)

def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]

def test_get_config():
    response = client.get("/config")
    assert response.status_code == 200
    data = response.json()
    assert "laser" in data

def test_train_status_initial():
    response = client.get("/train/status")
    assert response.status_code == 200
    data = response.json()
    assert data["is_running"] == False
    assert data["message"] == "Idle"
