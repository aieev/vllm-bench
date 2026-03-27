import openai
import pytest

IMAGE_URL = "https://images.pexels.com/photos/45201/kitty-cat-kitten-pet-45201.jpeg?auto=compress&cs=tinysrgb&w=400"


@pytest.mark.vision
def test_image_base64_color(client, model_name, extra_body, tiny_png_b64):
    data_url = f"data:image/png;base64,{tiny_png_b64}"
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
        extra_body=extra_body,
    )
    content = resp.choices[0].message.content or ""
    assert len(content.strip()) > 0, "Empty response for base64 image"


@pytest.mark.vision
def test_image_base64_pattern(client, model_name, extra_body, checkerboard_png_b64):
    data_url = f"data:image/png;base64,{checkerboard_png_b64}"
    resp = client.chat.completions.create(
        model=model_name,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe the pattern in this image."},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }],
        max_tokens=128,
        temperature=0.0,
        extra_body=extra_body,
    )
    content = resp.choices[0].message.content or ""
    assert len(content.strip()) > 10, f"Response too short for pattern description: {content}"


@pytest.mark.vision
def test_image_url_description(client, model_name, extra_body):
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
        extra_body=extra_body,
    )
    content = resp.choices[0].message.content or ""
    assert len(content.strip()) > 10, f"Response too short: {content}"


@pytest.mark.vision
def test_image_limit_exceeded(client, model_name, extra_body, tiny_png_b64):
    data_url = f"data:image/png;base64,{tiny_png_b64}"
    with pytest.raises(openai.BadRequestError):
        client.chat.completions.create(
            model=model_name,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Compare these images."},
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }],
            max_tokens=64,
            temperature=0.0,
            extra_body=extra_body,
        )


@pytest.mark.video
def test_video_description(client, model_name, extra_body):
    resp = client.chat.completions.create(
        model=model_name,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this video briefly."},
                {"type": "video_url", "video_url": {"url": "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerEscapes.mp4"}},
            ],
        }],
        max_tokens=128,
        temperature=0.0,
        extra_body=extra_body,
    )
    content = resp.choices[0].message.content or ""
    assert len(content.strip()) > 0, "Empty response for video"
