from __future__ import annotations

import io
from typing import Any

import httpx
from fastapi.testclient import TestClient

import mineru_adapter.default_proxy as default_proxy
from mineru_adapter.default_proxy import DefaultProxySettings, apply_default_fields, create_app


def test_apply_default_fields_adds_missing_defaults() -> None:
    settings = DefaultProxySettings(default_backend="vlm-http-client", default_server_url="http://adapter:18000")

    fields, decision = apply_default_fields([("return_md", "true")], settings)

    assert fields == [
        ("return_md", "true"),
        ("backend", "vlm-http-client"),
        ("server_url", "http://adapter:18000"),
    ]
    assert decision.route == "default"
    assert decision.backend == "vlm-http-client"


def test_apply_default_fields_routes_text_pdf_to_pipeline_without_server_url() -> None:
    settings = DefaultProxySettings(default_backend="vlm-http-client", default_server_url="http://adapter:18000")

    fields, decision = apply_default_fields(
        [("return_md", "true")],
        settings,
        text_pdf_detected=True,
        text_pdf_checked=True,
    )

    assert fields == [
        ("return_md", "true"),
        ("backend", "pipeline"),
    ]
    assert decision.route == "text-pdf"
    assert decision.backend == "pipeline"
    assert decision.text_pdf_detected is True
    assert decision.text_pdf_checked is True


def test_apply_default_fields_does_not_add_server_url_for_pipeline() -> None:
    settings = DefaultProxySettings(default_backend="vlm-http-client", default_server_url="http://adapter:18000")

    fields, decision = apply_default_fields([("backend", "pipeline")], settings)

    assert fields == [("backend", "pipeline")]
    assert decision.route == "explicit-backend"
    assert decision.backend == "pipeline"


def test_apply_default_fields_preserves_existing_values() -> None:
    settings = DefaultProxySettings(default_backend="vlm-http-client", default_server_url="http://adapter:18000")

    fields, decision = apply_default_fields(
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
    assert decision.route == "explicit-backend"
    assert decision.backend == "pipeline"
    assert decision.server_url == "http://custom:18000"


def test_apply_default_fields_can_force_defaults() -> None:
    settings = DefaultProxySettings(
        default_backend="vlm-http-client",
        default_server_url="http://adapter:18000",
        force_defaults=True,
    )

    fields, decision = apply_default_fields(
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
    assert decision.route == "default"
    assert decision.backend == "vlm-http-client"


def test_default_proxy_injects_multipart_defaults() -> None:
    captured: dict[str, Any] = {}

    async def fake_forwarder(
        method: str,
        target_url: str,
        headers: dict[str, str],
        body: bytes | None,
        data: dict[str, str | list[str]] | None,
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
    assert response.headers["x-mineru-proxy-route"] == "default"
    assert response.headers["x-mineru-proxy-backend"] == "vlm-http-client"
    assert response.headers["x-mineru-proxy-server-url"] == "http://mineru-adapter:18000"
    assert captured["method"] == "POST"
    assert captured["target_url"] == "http://official-mineru:8000/file_parse?trace_id=abc"
    assert captured["body"] is None
    assert captured["data"] == {
        "return_md": "true",
        "backend": "vlm-http-client",
        "server_url": "http://mineru-adapter:18000",
    }
    assert captured["files"][0][0] == "files"
    assert captured["files"][0][1][0] == "sample.pdf"
    assert not isinstance(captured["files"][0][1][1], bytes)
    assert hasattr(captured["files"][0][1][1], "read")


def test_default_proxy_skips_text_pdf_detection_when_backend_is_explicit(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def fail_if_called(*args: Any, **kwargs: Any) -> bool:
        raise AssertionError("text-layer detection should be skipped for explicit backend")

    async def fake_forwarder(
        method: str,
        target_url: str,
        headers: dict[str, str],
        body: bytes | None,
        data: dict[str, str | list[str]] | None,
        files: list[tuple[str, tuple[str, Any, str]]] | None,
        settings: DefaultProxySettings,
    ) -> httpx.Response:
        captured["data"] = data
        return httpx.Response(200, json={"ok": True})

    monkeypatch.setattr(default_proxy, "_pdf_has_text_layer", fail_if_called)
    settings = DefaultProxySettings(default_backend="vlm-http-client", default_server_url="http://mineru-adapter:18000")
    client = TestClient(create_app(settings, forwarder=fake_forwarder))

    response = client.post(
        "/file_parse",
        data={"backend": "pipeline"},
        files={"files": ("sample.pdf", b"%PDF", "application/pdf")},
    )

    assert response.status_code == 200
    assert response.headers["x-mineru-proxy-route"] == "explicit-backend"
    assert response.headers["x-mineru-proxy-text-pdf-checked"] == "false"
    assert captured["data"] == {"backend": "pipeline"}


def test_pdf_text_layer_detection_restores_stream_position() -> None:
    stream = io.BytesIO(b"%PDF invalid test payload")
    stream.seek(5)

    detected = default_proxy._pdf_has_text_layer(stream, DefaultProxySettings())

    assert detected is False
    assert stream.tell() == 5
