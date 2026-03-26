import os

import pytest
from pathlib import Path
from openai import OpenAI


def _load_env():
    """Load .env without external dependency."""
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip().strip("'\""))


_load_env()


@pytest.fixture(scope="session")
def base_url():
    url = os.environ.get("VLLM_BASE_URL")
    if not url:
        pytest.skip("VLLM_BASE_URL not set")
    return url


@pytest.fixture(scope="session")
def api_key():
    return os.environ.get("VLLM_API_KEY", "no-key")


@pytest.fixture(scope="session")
def model_name():
    name = os.environ.get("MODEL_NAME")
    if not name:
        pytest.skip("MODEL_NAME not set")
    return name


@pytest.fixture(scope="session")
def client(base_url, api_key):
    return OpenAI(base_url=f"{base_url}/v1", api_key=api_key)
