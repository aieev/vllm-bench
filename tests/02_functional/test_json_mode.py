import json
import pytest


def test_json_response_format(client, model_name, extra_body):
    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "user", "content": "Return a JSON object with a key 'greeting' and value 'hello'."},
        ],
        response_format={"type": "json_object"},
        max_tokens=128,
        temperature=0.0,
        extra_body=extra_body,
    )
    content = resp.choices[0].message.content
    assert content is not None
    parsed = json.loads(content)
    assert isinstance(parsed, dict)


def test_json_schema_validation(client, model_name, extra_body):
    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": (
                    "Return a JSON object with exactly these keys: "
                    '"name" (string), "age" (integer), "city" (string). '
                    "Use any values you like."
                ),
            },
        ],
        response_format={"type": "json_object"},
        max_tokens=128,
        temperature=0.0,
        extra_body=extra_body,
    )
    content = resp.choices[0].message.content
    assert content is not None
    parsed = json.loads(content)
    assert "name" in parsed, f"Missing 'name' key in {parsed}"
    assert "age" in parsed, f"Missing 'age' key in {parsed}"
    assert "city" in parsed, f"Missing 'city' key in {parsed}"
