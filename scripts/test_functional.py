"""
Functional LLM Tests: Function Calling, Image, Video

vLLM 서버의 기능적 정확성을 검증하는 테스트 스크립트.

Usage:
  python scripts/test_functional.py
  python scripts/test_functional.py --suite function_calling
  python scripts/test_functional.py --suite image --verbose
"""
import argparse
import base64
import json
import os
import struct
import sys
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import openai


# ---------------------------------------------------------------------------
# Minimal .env loader (no external dependency)
# ---------------------------------------------------------------------------

def load_dotenv(path: str = ".env"):
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip("'\"")
        os.environ.setdefault(key, val)


# ---------------------------------------------------------------------------
# Test infrastructure
# ---------------------------------------------------------------------------

@dataclass
class TestResult:
    name: str
    suite: str
    status: str  # "pass", "fail", "skip"
    detail: str = ""
    error: Optional[str] = None
    request: Optional[dict] = None
    response: Optional[str] = None


@dataclass
class TestRunner:
    client: openai.OpenAI
    model: str
    timeout: float = 30.0
    verbose: bool = False
    results: list = field(default_factory=list)

    def record(self, result: TestResult):
        self.results.append(result)
        icon = {"pass": "\u2705", "fail": "\u274c", "skip": "\u23ed\ufe0f"}[result.status]
        print(f"  {icon} {result.name:<30} \u2014 {result.detail}")
        if self.verbose and result.error:
            print(f"      Error: {result.error}")
        if self.verbose and result.response:
            print(f"      Response: {result.response[:300]}")

    def summary(self) -> int:
        passed = sum(1 for r in self.results if r.status == "pass")
        failed = sum(1 for r in self.results if r.status == "fail")
        skipped = sum(1 for r in self.results if r.status == "skip")
        print()
        print("=" * 40)
        print(f"Results: {passed} passed, {failed} failed, {skipped} skipped")
        print("=" * 40)
        return 1 if failed > 0 else 0


# ---------------------------------------------------------------------------
# Function Calling Tests
# ---------------------------------------------------------------------------

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

TOOL_CREATE_CALENDAR_EVENT = {
    "type": "function",
    "function": {
        "name": "create_calendar_event",
        "description": "Create a calendar event",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Event title"},
                "date": {"type": "string", "description": "Event date and time"},
                "duration_minutes": {"type": "integer", "description": "Duration in minutes"},
            },
            "required": ["title", "date", "duration_minutes"],
        },
    },
}

FC_EXTRA_BODY = {"chat_template_kwargs": {"enable_thinking": False}}


def test_simple_function_call(runner: TestRunner):
    name = "simple_function_call"
    try:
        resp = runner.client.chat.completions.create(
            model=runner.model,
            messages=[{"role": "user", "content": "What's the weather in Seoul?"}],
            tools=[TOOL_GET_WEATHER],
            tool_choice="auto",
            max_tokens=256,
            temperature=0.0,
            extra_body=FC_EXTRA_BODY,
            timeout=runner.timeout,
        )
        msg = resp.choices[0].message
        raw = msg.model_dump_json() if msg else ""

        if not msg.tool_calls or len(msg.tool_calls) == 0:
            runner.record(TestResult(name, "function_calling", "fail",
                                     "No tool_calls in response",
                                     response=raw))
            return

        tc = msg.tool_calls[0]
        if tc.function.name != "get_weather":
            runner.record(TestResult(name, "function_calling", "fail",
                                     f"Expected get_weather, got {tc.function.name}",
                                     response=raw))
            return

        args = json.loads(tc.function.arguments)
        location = args.get("location", "")
        if "seoul" not in location.lower():
            runner.record(TestResult(name, "function_calling", "fail",
                                     f"Expected Seoul in location, got: {location}",
                                     response=raw))
            return

        runner.record(TestResult(name, "function_calling", "pass",
                                 f"get_weather(location={location})",
                                 response=raw))

    except Exception as e:
        runner.record(TestResult(name, "function_calling", "fail",
                                 "Exception", error=str(e)))


def test_multi_tool_selection(runner: TestRunner):
    name = "multi_tool_selection"
    try:
        resp = runner.client.chat.completions.create(
            model=runner.model,
            messages=[{"role": "user", "content": "Translate 'hello world' to Korean"}],
            tools=[TOOL_GET_WEATHER, TOOL_GET_STOCK_PRICE, TOOL_TRANSLATE_TEXT],
            tool_choice="auto",
            max_tokens=256,
            temperature=0.0,
            extra_body=FC_EXTRA_BODY,
            timeout=runner.timeout,
        )
        msg = resp.choices[0].message
        raw = msg.model_dump_json() if msg else ""

        if not msg.tool_calls or len(msg.tool_calls) == 0:
            runner.record(TestResult(name, "function_calling", "fail",
                                     "No tool_calls in response",
                                     response=raw))
            return

        tc = msg.tool_calls[0]
        if tc.function.name != "translate_text":
            runner.record(TestResult(name, "function_calling", "fail",
                                     f"Expected translate_text, got {tc.function.name}",
                                     response=raw))
            return

        args = json.loads(tc.function.arguments)
        if "text" not in args or "target_language" not in args:
            runner.record(TestResult(name, "function_calling", "fail",
                                     f"Missing required args: {list(args.keys())}",
                                     response=raw))
            return

        runner.record(TestResult(name, "function_calling", "pass",
                                 f"translate_text selected (args: {list(args.keys())})",
                                 response=raw))

    except Exception as e:
        runner.record(TestResult(name, "function_calling", "fail",
                                 "Exception", error=str(e)))


def test_structured_tool_output(runner: TestRunner):
    name = "structured_tool_output"
    try:
        resp = runner.client.chat.completions.create(
            model=runner.model,
            messages=[{"role": "user",
                       "content": "Schedule a team meeting for tomorrow at 2pm for 30 minutes"}],
            tools=[TOOL_CREATE_CALENDAR_EVENT],
            tool_choice="auto",
            max_tokens=256,
            temperature=0.0,
            extra_body=FC_EXTRA_BODY,
            timeout=runner.timeout,
        )
        msg = resp.choices[0].message
        raw = msg.model_dump_json() if msg else ""

        if not msg.tool_calls or len(msg.tool_calls) == 0:
            runner.record(TestResult(name, "function_calling", "fail",
                                     "No tool_calls in response",
                                     response=raw))
            return

        tc = msg.tool_calls[0]
        if tc.function.name != "create_calendar_event":
            runner.record(TestResult(name, "function_calling", "fail",
                                     f"Expected create_calendar_event, got {tc.function.name}",
                                     response=raw))
            return

        args = json.loads(tc.function.arguments)
        if "title" not in args:
            runner.record(TestResult(name, "function_calling", "fail",
                                     f"Missing title in args: {list(args.keys())}",
                                     response=raw))
            return

        duration = args.get("duration_minutes")
        if duration is None:
            runner.record(TestResult(name, "function_calling", "fail",
                                     f"Missing duration_minutes: {list(args.keys())}",
                                     response=raw))
            return

        runner.record(TestResult(name, "function_calling", "pass",
                                 f"create_calendar_event(title={args['title']}, "
                                 f"duration={duration})",
                                 response=raw))

    except Exception as e:
        runner.record(TestResult(name, "function_calling", "fail",
                                 "Exception", error=str(e)))


# ---------------------------------------------------------------------------
# Image Tests
# ---------------------------------------------------------------------------

IMAGE_URL = "https://images.pexels.com/photos/45201/kitty-cat-kitten-pet-45201.jpeg?auto=compress&cs=tinysrgb&w=400"


def _make_red_png_base64() -> str:
    """Generate a minimal 1x1 red PNG as base64."""
    def _chunk(chunk_type, data):
        c = chunk_type + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = _chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
    raw = zlib.compress(b"\x00\xff\x00\x00")  # filter=none, R=255, G=0, B=0
    idat = _chunk(b"IDAT", raw)
    iend = _chunk(b"IEND", b"")
    return base64.b64encode(sig + ihdr + idat + iend).decode()


RED_PNG_B64 = _make_red_png_base64()


def test_image_url_description(runner: TestRunner):
    name = "image_url_description"
    try:
        resp = runner.client.chat.completions.create(
            model=runner.model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image in one sentence."},
                    {"type": "image_url", "image_url": {"url": IMAGE_URL}},
                ],
            }],
            max_tokens=128,
            temperature=0.0,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            timeout=runner.timeout,
        )
        content = resp.choices[0].message.content or ""
        if len(content.strip()) > 10:
            runner.record(TestResult(name, "image", "pass",
                                     f"{len(content)} chars response",
                                     response=content))
        else:
            runner.record(TestResult(name, "image", "fail",
                                     f"Response too short: {len(content)} chars",
                                     response=content))

    except openai.BadRequestError as e:
        err = str(e)
        if "image" in err.lower() or "multimodal" in err.lower() or "mm" in err.lower():
            runner.record(TestResult(name, "image", "skip",
                                     "Image not supported by model",
                                     error=err))
        else:
            runner.record(TestResult(name, "image", "fail",
                                     "BadRequest", error=err))
    except Exception as e:
        runner.record(TestResult(name, "image", "fail",
                                 "Exception", error=str(e)))


def test_image_base64_description(runner: TestRunner):
    name = "image_base64_description"
    try:
        data_url = f"data:image/png;base64,{RED_PNG_B64}"
        resp = runner.client.chat.completions.create(
            model=runner.model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "What color is this image?"},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }],
            max_tokens=64,
            temperature=0.0,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            timeout=runner.timeout,
        )
        content = resp.choices[0].message.content or ""
        if len(content.strip()) > 0:
            runner.record(TestResult(name, "image", "pass",
                                     f"{len(content)} chars response",
                                     response=content))
        else:
            runner.record(TestResult(name, "image", "fail",
                                     "Empty response", response=content))

    except openai.BadRequestError as e:
        err = str(e)
        if "image" in err.lower() or "multimodal" in err.lower() or "mm" in err.lower():
            runner.record(TestResult(name, "image", "skip",
                                     "Image not supported by model",
                                     error=err))
        else:
            runner.record(TestResult(name, "image", "fail",
                                     "BadRequest", error=err))
    except Exception as e:
        runner.record(TestResult(name, "image", "fail",
                                 "Exception", error=str(e)))


def test_image_limit_exceeded(runner: TestRunner):
    name = "image_limit_exceeded"
    try:
        data_url = f"data:image/png;base64,{RED_PNG_B64}"
        resp = runner.client.chat.completions.create(
            model=runner.model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Compare these two images."},
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }],
            max_tokens=64,
            temperature=0.0,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            timeout=runner.timeout,
        )
        runner.record(TestResult(name, "image", "fail",
                                 "Expected rejection but request succeeded",
                                 response=resp.choices[0].message.content or ""))

    except openai.BadRequestError as e:
        err = str(e)
        if any(kw in err.lower() for kw in ("image", "limit", "at most")):
            runner.record(TestResult(name, "image", "pass",
                                     "Server correctly rejected excess images",
                                     error=err))
        else:
            runner.record(TestResult(name, "image", "fail",
                                     "BadRequest but unexpected message",
                                     error=err))
    except Exception as e:
        runner.record(TestResult(name, "image", "fail",
                                 "Exception", error=str(e)))


# ---------------------------------------------------------------------------
# Video Tests
# ---------------------------------------------------------------------------

VIDEO_URL = "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerEscapes.mp4"


def test_video_url_description(runner: TestRunner):
    name = "video_url_description"
    try:
        resp = runner.client.chat.completions.create(
            model=runner.model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this video briefly."},
                    {"type": "video_url", "video_url": {"url": VIDEO_URL}},
                ],
            }],
            max_tokens=128,
            temperature=0.0,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            timeout=runner.timeout,
        )
        content = resp.choices[0].message.content or ""
        if len(content.strip()) > 0:
            runner.record(TestResult(name, "video", "pass",
                                     f"{len(content)} chars response",
                                     response=content))
        else:
            runner.record(TestResult(name, "video", "fail",
                                     "Empty response", response=content))

    except (openai.BadRequestError, openai.UnprocessableEntityError) as e:
        err = str(e)
        if any(kw in err.lower() for kw in ("video", "multimodal", "mm", "not supported", "limit")):
            runner.record(TestResult(name, "video", "skip",
                                     "Video not supported by model config",
                                     error=err))
        else:
            runner.record(TestResult(name, "video", "fail",
                                     "BadRequest", error=err))
    except Exception as e:
        runner.record(TestResult(name, "video", "fail",
                                 "Exception", error=str(e)))


# ---------------------------------------------------------------------------
# Suite registry
# ---------------------------------------------------------------------------

SUITES = {
    "function_calling": [
        test_simple_function_call,
        test_multi_tool_selection,
        test_structured_tool_output,
    ],
    "image": [
        test_image_url_description,
        test_image_base64_description,
        test_image_limit_exceeded,
    ],
    "video": [
        test_video_url_description,
    ],
}

SUITE_LABELS = {
    "function_calling": "Function Calling",
    "image": "Image",
    "video": "Video",
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Functional LLM Tests")
    parser.add_argument("--base-url", help="Override VLLM_BASE_URL")
    parser.add_argument("--api-key", help="Override VLLM_API_KEY")
    parser.add_argument("--model", help="Override MODEL_NAME")
    parser.add_argument("--suite", default="all",
                        choices=["all", "function_calling", "image", "video"],
                        help="Test suite to run (default: all)")
    parser.add_argument("--timeout", type=float, default=30.0,
                        help="Per-request timeout in seconds (default: 30)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print full request/response details")
    args = parser.parse_args()

    load_dotenv()

    base_url = args.base_url or os.environ.get("VLLM_BASE_URL")
    api_key = args.api_key or os.environ.get("VLLM_API_KEY", "no-key")
    model = args.model or os.environ.get("MODEL_NAME")

    if not base_url:
        print("Error: VLLM_BASE_URL not set. Use --base-url or set in .env", file=sys.stderr)
        sys.exit(1)
    if not model:
        print("Error: MODEL_NAME not set. Use --model or set in .env", file=sys.stderr)
        sys.exit(1)

    client = openai.OpenAI(base_url=f"{base_url}/v1", api_key=api_key)
    runner = TestRunner(client=client, model=model, timeout=args.timeout, verbose=args.verbose)

    print()
    print("\U0001f9ea Functional LLM Tests")
    print("=" * 40)
    print(f"Model: {model} @ {base_url}")
    print("=" * 40)

    suites_to_run = list(SUITES.keys()) if args.suite == "all" else [args.suite]

    for suite_name in suites_to_run:
        print(f"\n[{SUITE_LABELS[suite_name]}]")
        for test_fn in SUITES[suite_name]:
            test_fn(runner)

    exit_code = runner.summary()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
