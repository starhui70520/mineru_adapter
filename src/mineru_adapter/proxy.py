from __future__ import annotations

import copy
import json
import time
import uuid
from pathlib import Path
from typing import Any

import httpx

from .config import Settings
from .layout import LayoutParseError, layout_json_to_mineru_tags, strip_markdown_fence
from .messages import MinerUTask, detect_task, downsample_data_url_images, first_image_size, rewrite_messages_for_task


ALLOWED_OPENAI_FIELDS = {
    "model",
    "messages",
    "temperature",
    "top_p",
    "max_tokens",
    "max_completion_tokens",
    "presence_penalty",
    "frequency_penalty",
}


def build_upstream_payload(request_body: dict[str, Any], settings: Settings) -> tuple[dict[str, Any], MinerUTask, tuple[int, int] | None]:
    messages = request_body.get("messages")
    if not isinstance(messages, list):
        raise ValueError("Request body must contain a messages list")

    task = detect_task(messages)
    image_size = first_image_size(messages)

    if settings.drop_unsupported_params:
        payload = {
            key: copy.deepcopy(value)
            for key, value in request_body.items()
            if key in ALLOWED_OPENAI_FIELDS and key != "messages"
        }
    else:
        payload = {key: copy.deepcopy(value) for key, value in request_body.items() if key != "messages"}

    payload["model"] = settings.upstream_model
    outbound_messages = rewrite_messages_for_task(messages, task)
    if task == MinerUTask.layout and settings.layout_max_image_side > 0:
        outbound_messages, optimized_image_size = downsample_data_url_images(
            outbound_messages,
            settings.layout_max_image_side,
            jpeg_quality=settings.layout_jpeg_quality,
        )
        image_size = optimized_image_size or image_size
    payload["messages"] = outbound_messages
    payload["stream"] = False
    _apply_task_token_limit(payload, task, settings)
    if settings.disable_upstream_thinking:
        chat_template_kwargs = payload.get("chat_template_kwargs")
        if not isinstance(chat_template_kwargs, dict):
            chat_template_kwargs = {}
        chat_template_kwargs["enable_thinking"] = False
        payload["chat_template_kwargs"] = chat_template_kwargs
    return payload, task, image_size


def rewrite_upstream_response(
    upstream_response: dict[str, Any],
    task: MinerUTask,
    image_size: tuple[int, int] | None,
    settings: Settings,
) -> tuple[dict[str, Any], str, str | None]:
    response = copy.deepcopy(upstream_response)
    message = _first_choice_message(response)
    raw_content = message.get("content") if isinstance(message.get("content"), str) else ""

    parse_error: str | None = None
    if task == MinerUTask.layout:
        try:
            content = layout_json_to_mineru_tags(raw_content, image_size=image_size)
        except LayoutParseError as exc:
            parse_error = str(exc)
            content = ""
    elif task in {MinerUTask.text, MinerUTask.table, MinerUTask.formula}:
        content = strip_markdown_fence(raw_content)
    else:
        content = raw_content

    message["content"] = content
    if settings.strip_reasoning:
        message.pop("reasoning", None)
        message.pop("reasoning_content", None)
    response["model"] = settings.upstream_model
    return response, raw_content, parse_error


async def call_upstream(
    payload: dict[str, Any],
    settings: Settings,
    client: httpx.AsyncClient | None = None,
) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    if client is None:
        async with httpx.AsyncClient(timeout=settings.request_timeout) as local_client:
            response = await local_client.post(f"{settings.upstream_base_url}/v1/chat/completions", json=payload)
    else:
        response = await client.post(f"{settings.upstream_base_url}/v1/chat/completions", json=payload)
    elapsed = time.perf_counter() - started
    response.raise_for_status()
    return response.json(), elapsed


def _apply_task_token_limit(payload: dict[str, Any], task: MinerUTask, settings: Settings) -> None:
    limits = {
        MinerUTask.layout: settings.layout_max_tokens,
        MinerUTask.text: settings.text_max_tokens,
        MinerUTask.table: settings.table_max_tokens,
        MinerUTask.formula: settings.formula_max_tokens,
        MinerUTask.image: settings.image_max_tokens,
    }
    limit = limits.get(task)
    if limit is None or limit <= 0:
        return
    _cap_token_field(payload, "max_tokens", limit)
    _cap_token_field(payload, "max_completion_tokens", limit)
    if "max_tokens" not in payload and "max_completion_tokens" not in payload:
        payload["max_tokens"] = limit


def _cap_token_field(payload: dict[str, Any], field: str, limit: int) -> None:
    value = payload.get(field)
    if isinstance(value, (int, float)):
        payload[field] = min(int(value), limit)


def openai_models_response(settings: Settings) -> dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {
                "id": settings.upstream_model,
                "object": "model",
                "created": 0,
                "owned_by": "mineru-adapter",
            }
        ],
    }


def write_debug_record(
    settings: Settings,
    request_body: dict[str, Any],
    outbound_payload: dict[str, Any],
    upstream_response: dict[str, Any] | None,
    rewritten_response: dict[str, Any] | None,
    task: MinerUTask,
    elapsed_seconds: float | None,
    raw_content: str | None,
    parse_error: str | None,
    cache_status: str | None = None,
    exception: str | None = None,
) -> None:
    if settings.debug_dir is None:
        return
    settings.debug_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "id": str(uuid.uuid4()),
        "task": task.value,
        "elapsed_seconds": elapsed_seconds,
        "cache_status": cache_status,
        "parse_error": parse_error,
        "exception": exception,
        "request": _redact_large_images(request_body),
        "outbound_payload": _redact_large_images(outbound_payload),
        "upstream_raw_content": raw_content,
        "upstream_response": _redact_large_images(upstream_response),
        "rewritten_content": _safe_content(rewritten_response),
        "rewritten_response": _redact_large_images(rewritten_response),
    }
    target = settings.debug_dir / f"{int(time.time())}-{record['id']}.json"
    target.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")


def _first_choice_message(response: dict[str, Any]) -> dict[str, Any]:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("Upstream response does not contain choices")
    first = choices[0]
    if not isinstance(first, dict):
        raise ValueError("Upstream response choice is not an object")
    message = first.get("message")
    if not isinstance(message, dict):
        raise ValueError("Upstream response choice does not contain a message object")
    return message


def _safe_content(response: dict[str, Any] | None) -> str | None:
    if response is None:
        return None
    try:
        message = _first_choice_message(response)
    except ValueError:
        return None
    content = message.get("content")
    return content if isinstance(content, str) else None


def _redact_large_images(value: Any) -> Any:
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if key == "url" and isinstance(item, str) and item.startswith("data:image/"):
                result[key] = item[:80] + f"...<redacted {len(item)} chars>"
            else:
                result[key] = _redact_large_images(item)
        return result
    if isinstance(value, list):
        return [_redact_large_images(item) for item in value]
    return value
