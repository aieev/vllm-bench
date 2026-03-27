import os
import json
import base64
import struct
import zlib

import httpx
import pytest
import yaml
from pathlib import Path
from openai import OpenAI

PROBE_TIMEOUT = httpx.Timeout(60.0, connect=10.0)


def _load_env():
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

ROOT = Path(__file__).resolve().parent.parent


def _load_endpoints():
    config_path = ROOT / "endpoints.yaml"
    if not config_path.exists():
        return None, []
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg.get("defaults", {}), cfg.get("endpoints", [])


def _resolve_api_key(ep):
    env_var = ep.get("api_key_env", "VLLM_API_KEY")
    key = os.environ.get(env_var)
    if not key:
        key = os.environ.get("VLLM_API_KEY", "no-key")
    return key


def _make_tiny_png_base64():
    def _chunk(ct, data):
        c = ct + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = _chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
    idat = _chunk(b"IDAT", zlib.compress(b"\x00\xff\x00\x00"))
    iend = _chunk(b"IEND", b"")
    return base64.b64encode(sig + ihdr + idat + iend).decode()


_TINY_PNG = _make_tiny_png_base64()


def _make_checkerboard_png_base64(size=32, block=8):
    def _chunk(ct, data):
        c = ct + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

    raw_rows = []
    for y in range(size):
        row = b"\x00"
        for x in range(size):
            if ((x // block) + (y // block)) % 2 == 0:
                row += b"\xff\x00\x00"
            else:
                row += b"\x00\x00\xff"
        raw_rows.append(row)

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = _chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
    idat = _chunk(b"IDAT", zlib.compress(b"".join(raw_rows)))
    iend = _chunk(b"IEND", b"")
    return base64.b64encode(sig + ihdr + idat + iend).decode()


_CHECKERBOARD_PNG = _make_checkerboard_png_base64()


def _extra_body_for(is_thinking):
    if is_thinking:
        return {"chat_template_kwargs": {"enable_thinking": False}}
    return {}


def _detect_thinking(client, model_id):
    try:
        resp = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": "Say hi."}],
            max_tokens=16,
            temperature=0.0,
            timeout=PROBE_TIMEOUT,
        )
        msg = resp.choices[0].message
        raw = msg.model_dump() if msg else {}
        if raw.get("reasoning") is not None:
            return True
    except Exception:
        pass
    return False


def _detect_capabilities(client, model_id, is_thinking):
    caps = []
    eb = _extra_body_for(is_thinking)

    if is_thinking:
        caps.append("thinking")

    try:
        resp = client.chat.completions.create(
            model=model_id,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "What color?"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_TINY_PNG}"}},
                ],
            }],
            max_tokens=5,
            temperature=0.0,
            extra_body=eb,
            timeout=PROBE_TIMEOUT,
        )
        if resp.choices[0].message.content:
            caps.append("vision")
    except Exception:
        pass

    try:
        resp = client.chat.completions.create(
            model=model_id,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe."},
                    {"type": "video_url", "video_url": {"url": "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerEscapes.mp4"}},
                ],
            }],
            max_tokens=5,
            temperature=0.0,
            extra_body=eb,
            timeout=PROBE_TIMEOUT,
        )
        if resp.choices[0].message.content:
            caps.append("video")
    except Exception:
        pass

    try:
        tool = {
            "type": "function",
            "function": {
                "name": "test_probe",
                "description": "Test probe",
                "parameters": {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]},
            },
        }
        resp = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": "Call test_probe with x='hello'"}],
            tools=[tool],
            tool_choice="auto",
            max_tokens=64,
            temperature=0.0,
            extra_body=eb,
            timeout=PROBE_TIMEOUT,
        )
        if resp.choices[0].message.tool_calls:
            caps.append("function_calling")
    except Exception:
        pass

    return caps


def _get_active_endpoints():
    target = os.environ.get("TEST_ENDPOINT")
    defaults, endpoints = _load_endpoints()

    if not endpoints:
        return defaults, []

    if target:
        endpoints = [ep for ep in endpoints if ep["name"] == target]

    return defaults, endpoints


_DEFAULTS, _ENDPOINTS = _get_active_endpoints()
_DETECTED_CAPS = {}


def _detect_all_capabilities():
    for ep in _ENDPOINTS:
        name = ep["name"]
        if name in _DETECTED_CAPS:
            continue
        api_key = _resolve_api_key(ep)
        client = OpenAI(base_url=f"{ep['base_url']}/v1", api_key=api_key, timeout=PROBE_TIMEOUT)
        models = client.models.list()
        model_id = models.data[0].id if models.data else None
        if not model_id:
            _DETECTED_CAPS[name] = []
            continue
        print(f"  Probing {name}...", end="", flush=True)
        is_thinking = _detect_thinking(client, model_id)
        caps = _detect_capabilities(client, model_id, is_thinking)
        _DETECTED_CAPS[name] = caps
        print(f" {caps}")


if _ENDPOINTS:
    _detect_all_capabilities()


def pytest_configure(config):
    config.addinivalue_line("markers", "vision: requires vision capability")
    config.addinivalue_line("markers", "video: requires video capability")
    config.addinivalue_line("markers", "function_calling: requires function calling capability")


def _get_endpoint_name_from_item(item):
    for marker in item.iter_markers("parametrize"):
        if "endpoint" in marker.args[0] if isinstance(marker.args[0], str) else False:
            idx = item.callspec.indices.get("endpoint", 0)
            if idx < len(_ENDPOINTS):
                return _ENDPOINTS[idx]["name"]
    if _ENDPOINTS:
        return _ENDPOINTS[0]["name"]
    return ""


def pytest_collection_modifyitems(config, items):
    for item in items:
        ep_name = ""
        if hasattr(item, "callspec") and "endpoint" in item.callspec.params:
            ep = item.callspec.params["endpoint"]
            ep_name = ep["name"] if isinstance(ep, dict) else str(ep)
        elif _ENDPOINTS:
            ep_name = _ENDPOINTS[0]["name"]

        ep_caps = _DETECTED_CAPS.get(ep_name, [])

        for marker in ("vision", "video", "function_calling"):
            if marker in item.keywords and marker not in ep_caps:
                item.add_marker(pytest.mark.skip(
                    reason=f"Endpoint '{ep_name}' lacks {marker} capability"
                ))


def pytest_generate_tests(metafunc):
    if "endpoint" in metafunc.fixturenames:
        if _ENDPOINTS:
            metafunc.parametrize(
                "endpoint", _ENDPOINTS,
                ids=[ep["name"] for ep in _ENDPOINTS],
                scope="session",
            )
        else:
            pytest.skip("No endpoints configured")


@pytest.fixture(scope="session")
def endpoint():
    if _ENDPOINTS:
        return _ENDPOINTS[0]
    pytest.skip("No endpoints configured")


@pytest.fixture(scope="session")
def base_url(endpoint):
    return endpoint["base_url"]


@pytest.fixture(scope="session")
def api_key(endpoint):
    return _resolve_api_key(endpoint)


@pytest.fixture(scope="session")
def endpoint_name(endpoint):
    return endpoint["name"]


@pytest.fixture(scope="session")
def client(base_url, api_key):
    return OpenAI(base_url=f"{base_url}/v1", api_key=api_key)


@pytest.fixture(scope="session")
def model_name(client):
    name = os.environ.get("MODEL_NAME")
    if name:
        return name
    models = client.models.list()
    if models.data:
        return models.data[0].id
    pytest.skip("No model found via /v1/models")


@pytest.fixture(scope="session")
def capabilities(endpoint):
    return _DETECTED_CAPS.get(endpoint["name"], [])


@pytest.fixture(scope="session")
def extra_body(endpoint):
    caps = _DETECTED_CAPS.get(endpoint["name"], [])
    return _extra_body_for("thinking" in caps)


@pytest.fixture(scope="session")
def tiny_png_b64():
    return _TINY_PNG


@pytest.fixture(scope="session")
def checkerboard_png_b64():
    return _CHECKERBOARD_PNG
