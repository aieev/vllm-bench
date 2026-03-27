import pytest


def test_multi_turn_context(client, model_name, extra_body):
    messages = [
        {"role": "user", "content": "My name is Alice."},
    ]
    resp = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=32,
        temperature=0.0,
        extra_body=extra_body,
    )
    assistant_reply = resp.choices[0].message.content or ""
    messages.append({"role": "assistant", "content": assistant_reply})
    messages.append({"role": "user", "content": "What is my name?"})

    resp2 = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=32,
        temperature=0.0,
        extra_body=extra_body,
    )
    content = resp2.choices[0].message.content or ""
    assert "alice" in content.lower(), f"Model forgot the name. Response: {content}"


def test_multi_turn_system_persistence(client, model_name, extra_body):
    messages = [
        {"role": "system", "content": "You are a pirate. Always respond in pirate speak."},
        {"role": "user", "content": "Hello!"},
    ]
    resp = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=64,
        temperature=0.0,
        extra_body=extra_body,
    )
    first_reply = resp.choices[0].message.content or ""
    messages.append({"role": "assistant", "content": first_reply})
    messages.append({"role": "user", "content": "Tell me about the weather."})

    resp2 = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=64,
        temperature=0.0,
        extra_body=extra_body,
    )
    content = resp2.choices[0].message.content or ""
    assert len(content.strip()) > 0, "Empty response in multi-turn with system message"


def test_multi_turn_mid_length(client, model_name, extra_body):
    facts = {
        "color": "blue",
        "city": "Tokyo",
        "number": "42",
    }
    messages = [
        {"role": "user", "content": f"Remember: my favorite color is {facts['color']}."},
    ]

    def _chat(user_msg):
        resp = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=64,
            temperature=0.0,
            extra_body=extra_body,
        )
        reply = resp.choices[0].message.content or ""
        messages.append({"role": "assistant", "content": reply})
        messages.append({"role": "user", "content": user_msg})
        return reply

    _chat(f"Also remember: I live in {facts['city']}.")
    _chat("What do you think about Python programming?")
    _chat(f"One more thing: my lucky number is {facts['number']}.")
    _chat("Do you like music?")

    resp = client.chat.completions.create(
        model=model_name,
        messages=messages + [
            {"role": "user", "content": "List all three facts I told you: my favorite color, my city, and my lucky number."}
        ],
        max_tokens=128,
        temperature=0.0,
        extra_body=extra_body,
    )
    content = resp.choices[0].message.content or ""
    content_lower = content.lower()

    assert facts["color"] in content_lower, f"Forgot color '{facts['color']}'. Response: {content}"
    assert facts["city"].lower() in content_lower, f"Forgot city '{facts['city']}'. Response: {content}"
    assert facts["number"] in content, f"Forgot number '{facts['number']}'. Response: {content}"


def test_multi_turn_long_conversation(client, model_name, extra_body):
    secret_early = "PINEAPPLE"
    secret_mid = "TELESCOPE"
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Keep your answers brief."},
        {"role": "user", "content": f"The secret word for today is: {secret_early}. Please remember it."},
    ]

    filler_topics = [
        "What is the capital of France?",
        "Explain gravity in one sentence.",
        "Name three programming languages.",
        "What color is the sky?",
        "How many legs does a spider have?",
        "What is the boiling point of water?",
        "Name a famous scientist.",
        f"Here's another secret word to remember: {secret_mid}. Got it?",
        "What year did World War 2 end?",
        "What is the largest ocean?",
    ]

    for topic in filler_topics:
        resp = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=64,
            temperature=0.0,
            extra_body=extra_body,
        )
        reply = resp.choices[0].message.content or ""
        messages.append({"role": "assistant", "content": reply})
        messages.append({"role": "user", "content": topic})

    messages.append({"role": "user", "content": "What were the two secret words I told you earlier?"})
    resp = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=64,
        temperature=0.0,
        extra_body=extra_body,
    )
    content = resp.choices[0].message.content or ""
    content_upper = content.upper()

    assert secret_early in content_upper, (
        f"Lost early secret '{secret_early}' after 20+ messages. Response: {content}"
    )
    assert secret_mid in content_upper, (
        f"Lost mid secret '{secret_mid}' after 20+ messages. Response: {content}"
    )
