import requests
import pytest


def test_health_endpoint(base_url, api_key):
    resp = requests.get(
        f"{base_url}/health",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=10,
    )
    assert resp.status_code == 200


def test_models_endpoint(base_url, api_key):
    resp = requests.get(
        f"{base_url}/v1/models",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=10,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "data" in data
    assert isinstance(data["data"], list)
    assert len(data["data"]) > 0


def test_model_has_valid_config(base_url, api_key, model_name):
    resp = requests.get(
        f"{base_url}/v1/models",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=10,
    )
    models = resp.json()["data"]
    model = next((m for m in models if m["id"] == model_name), None)
    assert model is not None, f"Model {model_name} not found"
    assert model.get("max_model_len") and model["max_model_len"] > 0, "max_model_len should be positive"


def test_version_endpoint(base_url, api_key):
    resp = requests.get(
        f"{base_url}/version",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=10,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "version" in data
