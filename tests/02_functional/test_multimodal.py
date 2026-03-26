import base64
import struct
import zlib

import openai
import pytest

IMAGE_URL = "https://images.pexels.com/photos/45201/kitty-cat-kitten-pet-45201.jpeg?auto=compress&cs=tinysrgb&w=400"
VIDEO_URL = "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerEscapes.mp4"


def _make_red_png_base64() -> str:
    def _chunk(chunk_type, data):
        c = chunk_type + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = _chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
    raw = zlib.compress(b"\x00\xff\x00\x00")
    idat = _chunk(b"IDAT", raw)
    iend = _chunk(b"IEND", b"")
    return base64.b64encode(sig + ihdr + idat + iend).decode()


RED_PNG_B64 = _make_red_png_base64()

MM_EXTRA = {"chat_template_kwargs": {"enable_thinking": False}}


def test_image_url(client, model_name):
    resp = client.chat.completions.create(
        model=model_name,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image in one sentence."},
                {"type": "image_url", "image_url": {"url": IMAGE_URL}},
            ],
        }],
        max_tokens=128,
        temperature=0.0,
        extra_body=MM_EXTRA,
    )
    content = resp.choices[0].message.content or ""
    assert len(content.strip()) > 10, f"Response too short: {content}"


def test_image_base64(client, model_name):
    data_url = f"data:image/png;base64,{RED_PNG_B64}"
    resp = client.chat.completions.create(
        model=model_name,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "What color is this image?"},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }],
        max_tokens=64,
        temperature=0.0,
        extra_body=MM_EXTRA,
    )
    content = resp.choices[0].message.content or ""
    assert len(content.strip()) > 0, "Empty response for base64 image"


def test_image_limit_exceeded(client, model_name):
    data_url = f"data:image/png;base64,{RED_PNG_B64}"
    with pytest.raises(openai.BadRequestError) as exc_info:
        client.chat.completions.create(
            model=model_name,
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
            extra_body=MM_EXTRA,
        )
    err = str(exc_info.value).lower()
    assert any(kw in err for kw in ("image", "limit", "at most")), (
        f"Expected image limit error, got: {exc_info.value}"
    )


def test_video_not_supported(client, model_name):
    with pytest.raises((openai.BadRequestError, openai.UnprocessableEntityError)) as exc_info:
        client.chat.completions.create(
            model=model_name,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this video briefly."},
                    {"type": "video_url", "video_url": {"url": VIDEO_URL}},
                ],
            }],
            max_tokens=128,
            temperature=0.0,
            extra_body=MM_EXTRA,
        )
    err = str(exc_info.value).lower()
    assert any(kw in err for kw in ("video", "limit", "not supported", "mm")), (
        f"Expected video-related error, got: {exc_info.value}"
    )
