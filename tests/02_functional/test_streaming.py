import json

import httpx
import pytest


def test_sse_streaming(base_url, api_key, model_name, extra_body):
    chunks = []
    has_done = False

    with httpx.Client(timeout=30) as http:
        with http.stream(
            "POST",
            f"{base_url}/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model_name,
                "messages": [{"role": "user", "content": "Say hello."}],
                "stream": True,
                "max_tokens": 32,
                "temperature": 0.0,
                **extra_body,
            },
        ) as resp:
            assert resp.status_code == 200
            for line in resp.iter_lines():
                if not line:
                    continue
                assert line.startswith("data: "), f"Expected SSE format, got: {line}"
                payload = line[len("data: "):]
                if payload.strip() == "[DONE]":
                    has_done = True
                    break
                chunks.append(payload)

    assert len(chunks) > 0, "No streaming chunks received"
    assert has_done, "Stream did not end with [DONE]"


def test_streaming_token_count(base_url, api_key, model_name, extra_body):
    content_chunks = 0

    with httpx.Client(timeout=30) as http:
        with http.stream(
            "POST",
            f"{base_url}/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model_name,
                "messages": [{"role": "user", "content": "Count from 1 to 10."}],
                "stream": True,
                "max_tokens": 64,
                "temperature": 0.0,
                **extra_body,
            },
        ) as resp:
            for line in resp.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                payload = line[len("data: "):].strip()
                if payload == "[DONE]":
                    break
                chunk = json.loads(payload)
                choices = chunk.get("choices", [])
                if not choices:
                    continue
                delta = choices[0].get("delta", {})
                if delta.get("content"):
                    content_chunks += 1

    assert content_chunks > 0, "No content chunks in stream"


def test_streaming_vs_non_streaming(client, base_url, api_key, model_name, extra_body):
    non_stream = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": "What is 1+1?"}],
        max_tokens=16,
        temperature=0.0,
        extra_body=extra_body,
    )
    non_stream_content = non_stream.choices[0].message.content or ""

    stream_parts = []
    with httpx.Client(timeout=30) as http:
        with http.stream(
            "POST",
            f"{base_url}/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model_name,
                "messages": [{"role": "user", "content": "What is 1+1?"}],
                "stream": True,
                "max_tokens": 16,
                "temperature": 0.0,
                **extra_body,
            },
        ) as resp:
            for line in resp.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                payload = line[len("data: "):].strip()
                if payload == "[DONE]":
                    break
                chunk = json.loads(payload)
                choices = chunk.get("choices", [])
                if not choices:
                    continue
                delta = choices[0].get("delta", {})
                if delta.get("content"):
                    stream_parts.append(delta["content"])

    stream_content = "".join(stream_parts)
    assert len(non_stream_content) > 0, "Non-streaming response empty"
    assert len(stream_content) > 0, "Streaming response empty"
