from __future__ import annotations

import base64
import copy
import io

from PIL import Image
from mineru_adapter.config import Settings
from mineru_adapter.messages import MinerUTask
from mineru_adapter.proxy import build_upstream_payload, rewrite_upstream_response


def _request_body() -> dict:
    return {
        "model": "mineru-vl",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "\nLayout Detection:"}]}],
        "temperature": 0,
        "top_p": 0.01,
        "max_tokens": 1024,
        "vllm_xargs": {"no_repeat_ngram_size": 100},
        "skip_special_tokens": False,
        "priority": 1,
    }


def test_build_upstream_payload_filters_mineru_only_parameters() -> None:
    payload, task, _ = build_upstream_payload(_request_body(), Settings(upstream_model="vl-model"))

    assert task == MinerUTask.layout
    assert payload["model"] == "vl-model"
    assert "vllm_xargs" not in payload
    assert "skip_special_tokens" not in payload
    assert "priority" not in payload
    assert payload["stream"] is False
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert "JSON array" in payload["messages"][0]["content"][0]["text"]


def test_build_upstream_payload_can_preserve_extra_parameters_when_configured() -> None:
    body = _request_body()
    payload, _, _ = build_upstream_payload(body, Settings(upstream_model="vl-model", drop_unsupported_params=False))

    assert payload["vllm_xargs"] == {"no_repeat_ngram_size": 100}
    assert payload["skip_special_tokens"] is False
    assert payload["priority"] == 1
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}


def test_build_upstream_payload_can_keep_upstream_thinking_enabled() -> None:
    payload, _, _ = build_upstream_payload(
        _request_body(),
        Settings(upstream_model="vl-model", disable_upstream_thinking=False),
    )

    assert "chat_template_kwargs" not in payload


def test_build_upstream_payload_caps_layout_tokens() -> None:
    body = _request_body()
    body["max_tokens"] = 4096

    payload, _, _ = build_upstream_payload(body, Settings(upstream_model="vl-model", layout_max_tokens=512))

    assert payload["max_tokens"] == 512


def test_build_upstream_payload_downsamples_layout_images() -> None:
    body = _request_body()
    body["messages"][0]["content"].insert(0, {"type": "image_url", "image_url": {"url": _png_data_url(1200, 600)}})

    payload, task, image_size = build_upstream_payload(
        body,
        Settings(upstream_model="vl-model", layout_max_image_side=600),
    )

    assert task == MinerUTask.layout
    assert image_size == (600, 300)
    image_url = payload["messages"][0]["content"][0]["image_url"]["url"]
    assert _image_size_from_data_url(image_url) == (600, 300)
    assert image_url.startswith("data:image/png;base64,")


def test_build_upstream_payload_preserves_jpeg_layout_images() -> None:
    body = _request_body()
    body["messages"][0]["content"].insert(0, {"type": "image_url", "image_url": {"url": _jpeg_data_url(1200, 600)}})

    payload, task, image_size = build_upstream_payload(
        body,
        Settings(upstream_model="vl-model", layout_max_image_side=600, layout_jpeg_quality=85),
    )

    assert task == MinerUTask.layout
    assert image_size == (600, 300)
    image_url = payload["messages"][0]["content"][0]["image_url"]["url"]
    assert image_url.startswith("data:image/jpeg;base64,")
    assert _image_size_from_data_url(image_url) == (600, 300)


def test_rewrite_upstream_response_wraps_layout_as_openai_completion() -> None:
    upstream_response = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": "vl-model",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": '```json\n[{"bbox_2d":[1,2,3,4],"label":"title"}]\n```',
                    "reasoning": "hidden chain",
                    "reasoning_content": "hidden chain",
                },
                "finish_reason": "stop",
            }
        ],
    }

    rewritten, raw_content, parse_error = rewrite_upstream_response(
        copy.deepcopy(upstream_response),
        MinerUTask.layout,
        image_size=None,
        settings=Settings(strip_reasoning=True),
    )

    assert raw_content.startswith("```json")
    assert parse_error is None
    assert rewritten["choices"][0]["message"]["content"] == (
        "<|box_start|>1 2 3 4<|box_end|><|ref_start|>title<|ref_end|><|rotate_up|>"
    )
    assert "reasoning" not in rewritten["choices"][0]["message"]
    assert "reasoning_content" not in rewritten["choices"][0]["message"]


def test_rewrite_upstream_response_strips_markdown_for_text() -> None:
    response = {
        "choices": [
            {
                "message": {
                    "content": "```text\nExample Document\n```",
                }
            }
        ]
    }
    rewritten, _, _ = rewrite_upstream_response(response, MinerUTask.text, image_size=None, settings=Settings())

    assert rewritten["choices"][0]["message"]["content"] == "Example Document"


def _png_data_url(width: int, height: int) -> str:
    image = Image.new("RGB", (width, height), "white")
    output = io.BytesIO()
    image.save(output, format="PNG")
    return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode("ascii")


def _jpeg_data_url(width: int, height: int) -> str:
    image = Image.new("RGB", (width, height), "white")
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(output.getvalue()).decode("ascii")


def _image_size_from_data_url(url: str) -> tuple[int, int]:
    raw = base64.b64decode(url.split(",", 1)[1])
    with Image.open(io.BytesIO(raw)) as image:
        return image.size
