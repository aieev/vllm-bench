import pytest


def test_usage_fields_present(client, model_name, extra_body):
    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=16,
        temperature=0.0,
        extra_body=extra_body,
    )
    assert resp.usage is not None, "usage field missing from response"
    assert resp.usage.prompt_tokens > 0, "prompt_tokens should be > 0"
    assert resp.usage.completion_tokens > 0, "completion_tokens should be > 0"
    assert resp.usage.total_tokens == resp.usage.prompt_tokens + resp.usage.completion_tokens, (
        f"total_tokens ({resp.usage.total_tokens}) != prompt ({resp.usage.prompt_tokens}) + completion ({resp.usage.completion_tokens})"
    )


def test_prompt_token_count_scales(client, model_name, extra_body):
    short_resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": "Hi"}],
        max_tokens=5,
        temperature=0.0,
        extra_body=extra_body,
    )
    long_prompt = "Tell me everything you know about " + "the history of " * 50 + "computing."
    long_resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": long_prompt}],
        max_tokens=5,
        temperature=0.0,
        extra_body=extra_body,
    )
    assert long_resp.usage.prompt_tokens > short_resp.usage.prompt_tokens, (
        f"Longer prompt should have more tokens: short={short_resp.usage.prompt_tokens}, long={long_resp.usage.prompt_tokens}"
    )


def test_max_tokens_reflected_in_usage(client, model_name, extra_body):
    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": "Write a long story about a dragon."}],
        max_tokens=10,
        temperature=0.0,
        extra_body=extra_body,
    )
    assert resp.usage.completion_tokens <= 10, (
        f"completion_tokens ({resp.usage.completion_tokens}) exceeds max_tokens (10)"
    )
