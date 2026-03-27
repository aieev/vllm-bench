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

pytestmark = pytest.mark.function_calling


def test_simple_tool_call(client, model_name, extra_body):
    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": "What's the weather in Seoul?"}],
        tools=[TOOL_GET_WEATHER],
        tool_choice="auto",
        max_tokens=256,
        temperature=0.0,
        extra_body=extra_body,
    )
    msg = resp.choices[0].message
    assert msg.tool_calls and len(msg.tool_calls) > 0, "No tool_calls in response"

    tc = msg.tool_calls[0]
    assert tc.function.name == "get_weather", f"Expected get_weather, got {tc.function.name}"

    args = json.loads(tc.function.arguments)
    assert "location" in args, f"Missing 'location' in args: {args}"
    assert "seoul" in args["location"].lower(), f"Expected Seoul, got {args['location']}"


def test_multi_tool_selection(client, model_name, extra_body):
    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": "Translate 'hello world' to Korean"}],
        tools=[TOOL_GET_WEATHER, TOOL_GET_STOCK_PRICE, TOOL_TRANSLATE_TEXT],
        tool_choice="auto",
        max_tokens=256,
        temperature=0.0,
        extra_body=extra_body,
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


def test_no_tool_needed(client, model_name, extra_body):
    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": "What is 2+2?"}],
        tools=[TOOL_GET_WEATHER, TOOL_GET_STOCK_PRICE],
        tool_choice="auto",
        max_tokens=64,
        temperature=0.0,
        extra_body=extra_body,
    )
    msg = resp.choices[0].message
    assert not msg.tool_calls, (
        f"Model should not call tools for simple math, but called: {msg.tool_calls}"
    )
    assert msg.content and len(msg.content.strip()) > 0, "Expected content response"


TOOL_SEARCH = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "Search the web for information",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
            },
            "required": ["query"],
        },
    },
}

TOOL_CALCULATE = {
    "type": "function",
    "function": {
        "name": "calculate",
        "description": "Calculate a math expression and return the result",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Math expression to evaluate"},
            },
            "required": ["expression"],
        },
    },
}


def test_tool_output_faithfulness(client, model_name, extra_body):
    fake_population = "123,456"
    messages = [
        {"role": "user", "content": "Search for Iceland's population, then calculate 2% of it."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "search",
                    "arguments": json.dumps({"query": "Iceland population"}),
                },
            }],
        },
        {
            "role": "tool",
            "content": f"Iceland's current population is {fake_population}.",
            "tool_call_id": "call_1",
        },
    ]
    resp = client.chat.completions.create(
        model=model_name,
        messages=messages,
        tools=[TOOL_SEARCH, TOOL_CALCULATE],
        tool_choice="auto",
        max_tokens=256,
        temperature=0.0,
        extra_body=extra_body,
    )
    msg = resp.choices[0].message

    if msg.tool_calls:
        tc = msg.tool_calls[0]
        if tc.function.name == "calculate":
            expr = json.loads(tc.function.arguments).get("expression", "")
            assert "123456" in expr.replace(",", "").replace(" ", ""), (
                f"Model used its own knowledge instead of tool result. "
                f"Expected 123456 in expression, got: {expr}"
            )
            return

    content = (msg.content or "").replace(",", "").replace(" ", "")
    faithful = "2469" in content or "2469.12" in content.replace(",", "")
    unfaithful = any(x in content for x in ["7600", "7000", "7800", "3800"])
    assert not unfaithful, (
        f"Model ignored tool result and used its own knowledge. Response: {msg.content}"
    )
    assert faithful or "123456" in content, (
        f"Model didn't use the fake population (123,456). Response: {msg.content}"
    )


def test_chained_tool_calls(client, model_name, extra_body):
    messages = [
        {"role": "user", "content": "What is the weather in Tokyo in Fahrenheit?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": json.dumps({"location": "Tokyo"}),
                },
            }],
        },
        {
            "role": "tool",
            "content": json.dumps({"temperature": 17, "unit": "celsius", "condition": "cloudy"}),
            "tool_call_id": "call_1",
        },
    ]
    resp = client.chat.completions.create(
        model=model_name,
        messages=messages,
        tools=[TOOL_GET_WEATHER, TOOL_CALCULATE],
        tool_choice="auto",
        max_tokens=256,
        temperature=0.0,
        extra_body=extra_body,
    )
    msg = resp.choices[0].message
    content = msg.content or ""

    has_conversion = any(x in content for x in ["62", "63"])
    has_tool_call = False
    if msg.tool_calls:
        for tc in msg.tool_calls:
            if tc.function.name == "calculate":
                expr = json.loads(tc.function.arguments).get("expression", "")
                if "17" in expr:
                    has_tool_call = True

    assert has_conversion or has_tool_call, (
        f"Model should convert 17°C to ~62.6°F using tool data. "
        f"Response: {content}, tool_calls: {msg.tool_calls}"
    )
