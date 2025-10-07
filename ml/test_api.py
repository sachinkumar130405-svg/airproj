# tests/test_api.py
from fastapi.testclient import TestClient
import importlib, sys

# import your FastAPI app. Change `ml_api` to match your module if needed.
from ml import ml_api as app_module

client = TestClient(app_module.app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert "status" in r.json()

def test_root():
    r = client.get("/")
    assert r.status_code == 200
