from __future__ import annotations

import argparse
import io
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, TypeAlias

import httpx
from pypdf import PdfReader
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from starlette.datastructures import UploadFile


HOP_BY_HOP_HEADERS = {
    "connection",
    "host",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "content-length",
    "content-encoding",
}

FormDataValue: TypeAlias = str | list[str]
FormData: TypeAlias = dict[str, FormDataValue]


@dataclass(slots=True)
class ProxyDecision:
    backend: str | None = None
    server_url: str | None = None
    route: str = "passthrough"
    text_pdf_detected: bool = False
    text_pdf_checked: bool = False


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(slots=True)
class DefaultProxySettings:
    mineru_api_base_url: str = "http://official-mineru:8000"
    default_backend: str = "vlm-http-client"
    default_server_url: str = "http://mineru-adapter:18000"
    request_timeout: float = 600.0
    force_defaults: bool = False
    auto_text_pdf_routing: bool = True
    text_pdf_backend: str = "pipeline"
    text_pdf_min_chars: int = 120
    text_pdf_scan_pages: int = 3

    @classmethod
    def from_env(cls) -> "DefaultProxySettings":
        return cls(
            mineru_api_base_url=os.getenv("MINERU_API_BASE_URL", "http://official-mineru:8000").rstrip("/"),
            default_backend=os.getenv("DEFAULT_BACKEND", "vlm-http-client"),
            default_server_url=os.getenv("DEFAULT_SERVER_URL", "http://mineru-adapter:18000"),
            request_timeout=float(os.getenv("PROXY_REQUEST_TIMEOUT", "600")),
            force_defaults=_bool_env("FORCE_DEFAULTS", False),
            auto_text_pdf_routing=_bool_env("AUTO_TEXT_PDF_ROUTING", True),
            text_pdf_backend=os.getenv("TEXT_PDF_BACKEND", "pipeline"),
            text_pdf_min_chars=int(os.getenv("TEXT_PDF_MIN_CHARS", "120")),
            text_pdf_scan_pages=int(os.getenv("TEXT_PDF_SCAN_PAGES", "3")),
        )


Forwarder = Callable[
    [str, str, dict[str, str], bytes | None, FormData | None, list[tuple[str, tuple[str, Any, str]]] | None, DefaultProxySettings],
    Awaitable[httpx.Response],
]


def create_app(
    settings: DefaultProxySettings | None = None,
    forwarder: Forwarder | None = None,
) -> FastAPI:
    app_settings = settings or DefaultProxySettings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        async with httpx.AsyncClient(timeout=app_settings.request_timeout) as client:
            app.state.mineru_client = client
            yield

    app = FastAPI(title="MinerU Default Proxy", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "mineru_api_base_url": app_settings.mineru_api_base_url,
            "default_backend": app_settings.default_backend,
            "default_server_url": app_settings.default_server_url,
            "force_defaults": app_settings.force_defaults,
            "auto_text_pdf_routing": app_settings.auto_text_pdf_routing,
            "text_pdf_backend": app_settings.text_pdf_backend,
            "text_pdf_min_chars": app_settings.text_pdf_min_chars,
            "text_pdf_scan_pages": app_settings.text_pdf_scan_pages,
        }

    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"])
    async def proxy(path: str, request: Request) -> Response:
        target_url = build_target_url(app_settings.mineru_api_base_url, path, str(request.url.query))
        headers = filter_headers(dict(request.headers))

        body: bytes | None = None
        data: list[tuple[str, str]] | None = None
        files: list[tuple[str, tuple[str, Any, str]]] | None = None
        decision = ProxyDecision()

        if should_inject_defaults(path, request):
            data, files, decision = await build_multipart_payload(request, app_settings)
            headers.pop("content-type", None)
        else:
            body = await request.body()

        try:
            if forwarder is None:
                upstream_response = await forward_to_mineru(
                    request.method,
                    target_url,
                    headers,
                    body,
                    data,
                    files,
                    app_settings,
                    client=getattr(request.app.state, "mineru_client", None),
                )
            else:
                upstream_response = await forwarder(
                    request.method,
                    target_url,
                    headers,
                    body,
                    data,
                    files,
                    app_settings,
                )
        except httpx.RequestError as exc:
            raise HTTPException(status_code=502, detail=f"Failed to reach MinerU API: {exc}") from exc

        response_headers = filter_headers(dict(upstream_response.headers))
        response_headers.update(proxy_decision_headers(decision))
        return Response(
            content=upstream_response.content,
            status_code=upstream_response.status_code,
            headers=response_headers,
            media_type=upstream_response.headers.get("content-type"),
        )

    return app


def build_target_url(base_url: str, path: str, query: str) -> str:
    url = f"{base_url.rstrip('/')}/{path.lstrip('/')}"
    if query:
        return f"{url}?{query}"
    return url


def should_inject_defaults(path: str, request: Request) -> bool:
    normalized_path = path.strip("/")
    content_type = request.headers.get("content-type", "")
    return request.method.upper() == "POST" and normalized_path == "file_parse" and "multipart/form-data" in content_type


async def build_multipart_payload(
    request: Request,
    settings: DefaultProxySettings,
) -> tuple[FormData, list[tuple[str, tuple[str, Any, str]]], ProxyDecision]:
    form = await request.form()
    items = list(form.multi_items())
    fields: list[tuple[str, str]] = []
    files: list[tuple[str, tuple[str, Any, str]]] = []
    pdf_uploads = 0
    text_pdf_uploads = 0

    for key, value in items:
        if not isinstance(value, UploadFile):
            fields.append((key, str(value)))

    existing = {key for key, _ in fields}
    should_check_text_pdf = settings.auto_text_pdf_routing and (settings.force_defaults or "backend" not in existing)

    for key, value in items:
        if isinstance(value, UploadFile):
            filename = value.filename or "upload"
            content_type = value.content_type or "application/octet-stream"
            if should_check_text_pdf and _is_pdf_upload(filename, content_type):
                await value.seek(0)
                pdf_uploads += 1
                if _pdf_has_text_layer(value.file, settings):
                    text_pdf_uploads += 1
                await value.seek(0)
                file_body: Any = value.file
            else:
                await value.seek(0)
                file_body = value.file
            files.append(
                (
                    key,
                    (
                        filename,
                        file_body,
                        content_type,
                    ),
                )
            )

    text_pdf_detected = pdf_uploads > 0 and pdf_uploads == text_pdf_uploads
    applied_fields, decision = apply_default_fields(
        fields,
        settings,
        text_pdf_detected=text_pdf_detected,
        text_pdf_checked=pdf_uploads > 0,
    )
    return fields_to_httpx_data(applied_fields), files, decision


def apply_default_fields(
    fields: list[tuple[str, str]],
    settings: DefaultProxySettings,
    text_pdf_detected: bool = False,
    text_pdf_checked: bool = False,
) -> tuple[list[tuple[str, str]], ProxyDecision]:
    result = list(fields)
    existing = {key for key, _ in result}
    decision = ProxyDecision(
        backend=_last_field_value(result, "backend"),
        server_url=_last_field_value(result, "server_url"),
        text_pdf_detected=text_pdf_detected,
        text_pdf_checked=text_pdf_checked,
    )

    if settings.force_defaults:
        result = [(key, value) for key, value in result if key not in {"backend", "server_url"}]
        existing = {key for key, _ in result}
        decision.route = "forced-defaults"

    if "backend" not in existing:
        backend = settings.text_pdf_backend if settings.auto_text_pdf_routing and text_pdf_detected else settings.default_backend
        result.append(("backend", backend))
        existing.add("backend")
        decision.backend = backend
        decision.route = "text-pdf" if settings.auto_text_pdf_routing and text_pdf_detected else "default"
    else:
        backend = _last_field_value(result, "backend") or settings.default_backend
        decision.backend = backend
        if decision.route == "passthrough":
            decision.route = "explicit-backend"

    if _backend_requires_server_url(backend) and "server_url" not in existing:
        result.append(("server_url", settings.default_server_url))
        decision.server_url = settings.default_server_url
    else:
        decision.server_url = _last_field_value(result, "server_url")
    return result, decision


def fields_to_httpx_data(fields: list[tuple[str, str]]) -> FormData:
    data: FormData = {}
    for key, value in fields:
        existing = data.get(key)
        if existing is None:
            data[key] = value
        elif isinstance(existing, list):
            existing.append(value)
        else:
            data[key] = [existing, value]
    return data


async def forward_to_mineru(
    method: str,
    target_url: str,
    headers: dict[str, str],
    body: bytes | None,
    data: FormData | None,
    files: list[tuple[str, tuple[str, Any, str]]] | None,
    settings: DefaultProxySettings,
    client: httpx.AsyncClient | None = None,
) -> httpx.Response:
    if client is None:
        async with httpx.AsyncClient(timeout=settings.request_timeout) as local_client:
            return await local_client.request(
                method,
                target_url,
                headers=headers,
                content=body,
                data=data,
                files=files,
            )
    return await client.request(
        method,
        target_url,
        headers=headers,
        content=body,
        data=data,
        files=files,
    )


def filter_headers(headers: dict[str, str]) -> dict[str, str]:
    return {key: value for key, value in headers.items() if key.lower() not in HOP_BY_HOP_HEADERS}


def proxy_decision_headers(decision: ProxyDecision) -> dict[str, str]:
    headers = {
        "x-mineru-proxy-route": decision.route,
        "x-mineru-proxy-text-pdf": str(decision.text_pdf_detected).lower(),
        "x-mineru-proxy-text-pdf-checked": str(decision.text_pdf_checked).lower(),
    }
    if decision.backend:
        headers["x-mineru-proxy-backend"] = decision.backend
    if decision.server_url:
        headers["x-mineru-proxy-server-url"] = decision.server_url
    return headers


def _last_field_value(fields: list[tuple[str, str]], key: str) -> str | None:
    for field_key, value in reversed(fields):
        if field_key == key:
            return value
    return None


def _backend_requires_server_url(backend: str) -> bool:
    normalized = backend.strip().lower()
    return normalized.endswith("http-client")


def _is_pdf_upload(filename: str, content_type: str) -> bool:
    return filename.lower().endswith(".pdf") or content_type.lower() == "application/pdf"


def _pdf_has_text_layer(file_obj: Any, settings: DefaultProxySettings) -> bool:
    stream = io.BytesIO(file_obj) if isinstance(file_obj, bytes) else file_obj
    original_position: int | None = None
    try:
        original_position = stream.tell()
    except Exception:
        original_position = None

    try:
        stream.seek(0)
        header = stream.read(1024)
        if not isinstance(header, bytes) or not header.lstrip().startswith(b"%PDF"):
            return False
        stream.seek(0)
        reader = PdfReader(stream)
        return _pages_have_text_layer(reader.pages, settings.text_pdf_min_chars, settings.text_pdf_scan_pages)
    except Exception:
        return False
    finally:
        try:
            stream.seek(original_position or 0)
        except Exception:
            pass


def _pages_have_text_layer(pages: Any, min_chars: int, scan_pages: int) -> bool:
    max_pages = min(len(pages), max(1, scan_pages))
    nonspace_chars = 0
    for index in range(max_pages):
        nonspace_chars = _count_nonspace_chars(pages[index].extract_text() or "", nonspace_chars, min_chars)
        if nonspace_chars >= min_chars:
            return True
    return nonspace_chars >= min_chars


def _count_nonspace_chars(text: str, current: int = 0, limit: int | None = None) -> int:
    for char in text:
        if char.isspace():
            continue
        current += 1
        if limit is not None and current >= limit:
            return current
    return current


app = create_app()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the MinerU default proxy.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run("mineru_adapter.default_proxy:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
