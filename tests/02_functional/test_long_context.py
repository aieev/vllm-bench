import pytest


def test_long_context_8k(client, model_name, extra_body):
    long_text = "The quick brown fox jumps over the lazy dog. " * 200
    secret = "SECRET_CODE_12345"
    prompt = f"Here is a long document:\n\n{long_text}\n\nRemember this code: {secret}\n\nWhat is the code I asked you to remember?"

    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=32,
        temperature=0.0,
        extra_body=extra_body,
    )
    content = resp.choices[0].message.content or ""
    assert secret in content, f"Model failed to retrieve code from ~8K context. Response: {content}"


def test_long_context_does_not_crash(client, model_name, extra_body):
    long_text = "A " * 5000
    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": f"{long_text}\n\nSummarize the above in one word."}],
        max_tokens=16,
        temperature=0.0,
        extra_body=extra_body,
    )
    assert resp.choices[0].message.content is not None
    assert resp.usage.prompt_tokens > 1000, f"Expected >1000 prompt tokens, got {resp.usage.prompt_tokens}"
