import pytest


def test_chat_completion_basic(client, model_name):
    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=32,
        temperature=0.0,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    content = resp.choices[0].message.content
    assert content and len(content.strip()) > 0


def test_thinking_mode_disable(client, model_name):
    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": "Say hello."}],
        max_tokens=32,
        temperature=0.0,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    msg = resp.choices[0].message
    assert msg.content is not None and len(msg.content.strip()) > 0


def test_system_message(client, model_name):
    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say hi."},
        ],
        max_tokens=32,
        temperature=0.0,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    content = resp.choices[0].message.content
    assert content and len(content.strip()) > 0


def test_max_tokens_respected(client, model_name):
    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": "Count from 1 to 100."}],
        max_tokens=5,
        temperature=0.0,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    content = resp.choices[0].message.content or ""
    tokens = content.split()
    assert len(tokens) <= 10, f"Expected <=10 tokens, got {len(tokens)}: {content}"
