import openai
import pytest


def test_invalid_model_name(client, extra_body):
    with pytest.raises(openai.NotFoundError):
        client.chat.completions.create(
            model="nonexistent-model-xyz-999",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=16,
            extra_body=extra_body,
        )


def test_empty_messages(client, model_name, extra_body):
    with pytest.raises((openai.BadRequestError, openai.UnprocessableEntityError)):
        client.chat.completions.create(
            model=model_name,
            messages=[],
            max_tokens=16,
            extra_body=extra_body,
        )


def test_invalid_max_tokens(client, model_name, extra_body):
    with pytest.raises((openai.BadRequestError, openai.UnprocessableEntityError)):
        client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=-1,
            extra_body=extra_body,
        )


def test_invalid_temperature(client, model_name, extra_body):
    with pytest.raises((openai.BadRequestError, openai.UnprocessableEntityError)):
        client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=16,
            temperature=-1.0,
            extra_body=extra_body,
        )
