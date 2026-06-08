from __future__ import annotations

import asyncio
from typing import Any

from fastapi.testclient import TestClient

import mineru_adapter.api as api
from mineru_adapter.api import create_app, schedule_debug_record
from mineru_adapter.config import Settings


async def fake_upstream_caller(payload: dict[str, Any], settings: Settings):
    assert payload["model"] == "vl-model"
    return (
        {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "model": "vl-model",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": '[{"bbox_2d":[10,20,30,40],"label":"title"}]',
                    },
                    "finish_reason": "stop",
                }
            ],
        },
        0.01,
    )


def test_models_endpoint() -> None:
    client = TestClient(create_app(Settings(upstream_model="vl-model"), upstream_caller=fake_upstream_caller))

    response = client.get("/v1/models")

    assert response.status_code == 200
    assert response.json()["data"][0]["id"] == "vl-model"


def test_chat_completion_rewrites_layout_response() -> None:
    client = TestClient(create_app(Settings(upstream_model="vl-model"), upstream_caller=fake_upstream_caller))

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "vl-model",
            "messages": [{"role": "user", "content": "\nLayout Detection:"}],
            "vllm_xargs": {"no_repeat_ngram_size": 100},
        },
    )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == (
        "<|box_start|>10 20 30 40<|box_end|><|ref_start|>title<|ref_end|><|rotate_up|>"
    )


def test_schedule_debug_record_sync(monkeypatch, tmp_path) -> None:
    calls: list[tuple[Any, ...]] = []

    def fake_write_debug_record(*args: Any, **kwargs: Any) -> None:
        calls.append(args)

    monkeypatch.setattr(api, "write_debug_record", fake_write_debug_record)

    schedule_debug_record(Settings(debug_dir=tmp_path, debug_async=False), "request")

    assert len(calls) == 1
    assert calls[0][1] == "request"


def test_schedule_debug_record_async(monkeypatch, tmp_path) -> None:
    async def scenario() -> None:
        calls: list[tuple[Any, ...]] = []

        def fake_write_debug_record(*args: Any, **kwargs: Any) -> None:
            calls.append(args)

        monkeypatch.setattr(api, "write_debug_record", fake_write_debug_record)

        schedule_debug_record(Settings(debug_dir=tmp_path, debug_async=True), "request")
        await asyncio.sleep(0.05)

        assert len(calls) == 1
        assert calls[0][1] == "request"

    asyncio.run(scenario())
