import pytest


def test_stop_sequence(client, model_name, extra_body):
    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": "Write exactly this text: 'Hello World END Goodbye'",
            },
        ],
        stop=["END"],
        max_tokens=64,
        temperature=0.0,
        extra_body=extra_body,
    )
    content = resp.choices[0].message.content or ""
    assert "Goodbye" not in content, f"Stop sequence not respected: {content}"


def test_max_tokens_limit(client, model_name, extra_body):
    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "user", "content": "Write a very long essay about the history of computing."},
        ],
        max_tokens=10,
        temperature=0.0,
        extra_body=extra_body,
    )
    assert resp.choices[0].finish_reason == "length", (
        f"Expected finish_reason='length', got '{resp.choices[0].finish_reason}'"
    )
    content = resp.choices[0].message.content or ""
    assert len(content) > 0, "Response should not be empty"
