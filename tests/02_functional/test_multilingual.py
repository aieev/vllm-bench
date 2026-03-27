import pytest


def test_korean_input(client, model_name, extra_body):
    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": "안녕하세요. 오늘 날씨는 어떤가요?"}],
        max_tokens=64,
        temperature=0.0,
        extra_body=extra_body,
    )
    content = resp.choices[0].message.content or ""
    assert len(content.strip()) > 0, "Empty response for Korean input"


def test_chinese_input(client, model_name, extra_body):
    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": "你好，请用中文回答：1+1等于多少？"}],
        max_tokens=64,
        temperature=0.0,
        extra_body=extra_body,
    )
    content = resp.choices[0].message.content or ""
    assert len(content.strip()) > 0, "Empty response for Chinese input"


def test_japanese_input(client, model_name, extra_body):
    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": "こんにちは。今日の天気はどうですか？"}],
        max_tokens=64,
        temperature=0.0,
        extra_body=extra_body,
    )
    content = resp.choices[0].message.content or ""
    assert len(content.strip()) > 0, "Empty response for Japanese input"


def test_emoji_input(client, model_name, extra_body):
    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": "What does this emoji mean? 🚀🔥💯"}],
        max_tokens=64,
        temperature=0.0,
        extra_body=extra_body,
    )
    content = resp.choices[0].message.content or ""
    assert len(content.strip()) > 0, "Empty response for emoji input"


def test_mixed_language(client, model_name, extra_body):
    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": "Translate '감사합니다' to English."}],
        max_tokens=32,
        temperature=0.0,
        extra_body=extra_body,
    )
    content = resp.choices[0].message.content or ""
    assert "thank" in content.lower(), f"Expected 'thank' in translation, got: {content}"
