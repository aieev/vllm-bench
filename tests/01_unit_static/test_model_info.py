import requests
import pytest


def test_metrics_endpoint(base_url, api_key):
    resp = requests.get(
        f"{base_url}/metrics",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=10,
    )
    if resp.status_code == 404:
        pytest.skip("/metrics not exposed through gateway")
    assert resp.status_code == 200
    assert "vllm" in resp.text


def test_vram_allocation_in_metrics(base_url, api_key):
    resp = requests.get(
        f"{base_url}/metrics",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=10,
    )
    if resp.status_code == 404:
        pytest.skip("/metrics not exposed through gateway")
    text = resp.text
    has_gpu_metric = (
        "vllm:gpu_cache_usage_perc" in text
        or "gpu_cache_usage" in text
        or "vllm:num_requests" in text
    )
    assert has_gpu_metric, "Expected GPU-related metrics in /metrics"


def test_model_config(base_url, api_key, model_name):
    resp = requests.get(
        f"{base_url}/v1/models",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=10,
    )
    models = resp.json()["data"]
    model = next((m for m in models if m["id"] == model_name), None)
    assert model is not None, f"Model {model_name} not found"
    assert "id" in model
    assert "object" in model
