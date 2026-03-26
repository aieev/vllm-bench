import json
import pytest

TOOL_GET_WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["location"],
        },
    },
}

TOOL_TRANSLATE_TEXT = {
    "type": "function",
    "function": {
        "name": "translate_text",
        "description": "Translate text to a target language",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to translate"},
                "target_language": {"type": "string", "description": "Target language"},
            },
            "required": ["text", "target_language"],
        },
    },
}

TOOL_GET_STOCK_PRICE = {
    "type": "function",
    "function": {
        "name": "get_stock_price",
        "description": "Get the current stock price for a given ticker symbol",
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Stock ticker symbol"},
            },
            "required": ["symbol"],
        },
    },
}

FC_EXTRA = {"chat_template_kwargs": {"enable_thinking": False}}


def test_simple_tool_call(client, model_name):
    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": "What's the weather in Seoul?"}],
        tools=[TOOL_GET_WEATHER],
        tool_choice="auto",
        max_tokens=256,
        temperature=0.0,
        extra_body=FC_EXTRA,
    )
    msg = resp.choices[0].message
    assert msg.tool_calls and len(msg.tool_calls) > 0, "No tool_calls in response"

    tc = msg.tool_calls[0]
    assert tc.function.name == "get_weather", f"Expected get_weather, got {tc.function.name}"

    args = json.loads(tc.function.arguments)
    assert "location" in args, f"Missing 'location' in args: {args}"
    assert "seoul" in args["location"].lower(), f"Expected Seoul, got {args['location']}"


def test_multi_tool_selection(client, model_name):
    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": "Translate 'hello world' to Korean"}],
        tools=[TOOL_GET_WEATHER, TOOL_GET_STOCK_PRICE, TOOL_TRANSLATE_TEXT],
        tool_choice="auto",
        max_tokens=256,
        temperature=0.0,
        extra_body=FC_EXTRA,
    )
    msg = resp.choices[0].message
    assert msg.tool_calls and len(msg.tool_calls) > 0, "No tool_calls in response"

    tc = msg.tool_calls[0]
    assert tc.function.name == "translate_text", (
        f"Expected translate_text, got {tc.function.name}"
    )

    args = json.loads(tc.function.arguments)
    assert "text" in args, f"Missing 'text' in args: {args}"
    assert "target_language" in args, f"Missing 'target_language' in args: {args}"


def test_no_tool_needed(client, model_name):
    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": "What is 2+2?"}],
        tools=[TOOL_GET_WEATHER, TOOL_GET_STOCK_PRICE],
        tool_choice="auto",
        max_tokens=64,
        temperature=0.0,
        extra_body=FC_EXTRA,
    )
    msg = resp.choices[0].message
    has_content = msg.content and len(msg.content.strip()) > 0
    has_no_tools = not msg.tool_calls or len(msg.tool_calls) == 0
    assert has_content or has_no_tools, (
        "Model should respond with content or skip tools for simple math"
    )
