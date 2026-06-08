from __future__ import annotations

from typing import Any

import httpx
from fastapi.testclient import TestClient

from mineru_adapter.default_proxy import DefaultProxySettings, apply_default_fields, create_app


def test_apply_default_fields_adds_missing_defaults() -> None:
    settings = DefaultProxySettings(default_backend="vlm-http-client", default_server_url="http://adapter:18000")

    fields = apply_default_fields([("return_md", "true")], settings)

    assert fields == [
        ("return_md", "true"),
        ("backend", "vlm-http-client"),
        ("server_url", "http://adapter:18000"),
    ]


def test_apply_default_fields_preserves_existing_values() -> None:
    settings = DefaultProxySettings(default_backend="vlm-http-client", default_server_url="http://adapter:18000")

    fields = apply_default_fields(
        [
            ("backend", "pipeline"),
            ("server_url", "http://custom:18000"),
        ],
        settings,
    )

    assert fields == [
        ("backend", "pipeline"),
        ("server_url", "http://custom:18000"),
    ]


def test_apply_default_fields_can_force_defaults() -> None:
    settings = DefaultProxySettings(
        default_backend="vlm-http-client",
        default_server_url="http://adapter:18000",
        force_defaults=True,
    )

    fields = apply_default_fields(
        [
            ("backend", "pipeline"),
            ("server_url", "http://custom:18000"),
            ("return_md", "true"),
        ],
        settings,
    )

    assert fields == [
        ("return_md", "true"),
        ("backend", "vlm-http-client"),
        ("server_url", "http://adapter:18000"),
    ]


def test_default_proxy_injects_multipart_defaults() -> None:
    captured: dict[str, Any] = {}

    async def fake_forwarder(
        method: str,
        target_url: str,
        headers: dict[str, str],
        body: bytes | None,
        data: list[tuple[str, str]] | None,
        files: list[tuple[str, tuple[str, Any, str]]] | None,
        settings: DefaultProxySettings,
    ) -> httpx.Response:
        captured.update(
            {
                "method": method,
                "target_url": target_url,
                "body": body,
                "data": data,
                "files": files,
            }
        )
        return httpx.Response(200, json={"ok": True})

    settings = DefaultProxySettings(
        mineru_api_base_url="http://official-mineru:8000",
        default_backend="vlm-http-client",
        default_server_url="http://mineru-adapter:18000",
    )
    client = TestClient(create_app(settings, forwarder=fake_forwarder))

    response = client.post(
        "/file_parse?trace_id=abc",
        data={"return_md": "true"},
        files={"files": ("sample.pdf", b"%PDF", "application/pdf")},
    )

    assert response.status_code == 200
    assert captured["method"] == "POST"
    assert captured["target_url"] == "http://official-mineru:8000/file_parse?trace_id=abc"
    assert captured["body"] is None
    assert captured["data"] == [
        ("return_md", "true"),
        ("backend", "vlm-http-client"),
        ("server_url", "http://mineru-adapter:18000"),
    ]
    assert captured["files"][0][0] == "files"
    assert captured["files"][0][1][0] == "sample.pdf"
