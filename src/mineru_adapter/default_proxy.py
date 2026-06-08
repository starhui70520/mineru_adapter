from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from starlette.datastructures import UploadFile


HOP_BY_HOP_HEADERS = {
    "connection",
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

    @classmethod
    def from_env(cls) -> "DefaultProxySettings":
        return cls(
            mineru_api_base_url=os.getenv("MINERU_API_BASE_URL", "http://official-mineru:8000").rstrip("/"),
            default_backend=os.getenv("DEFAULT_BACKEND", "vlm-http-client"),
            default_server_url=os.getenv("DEFAULT_SERVER_URL", "http://mineru-adapter:18000"),
            request_timeout=float(os.getenv("PROXY_REQUEST_TIMEOUT", "600")),
            force_defaults=_bool_env("FORCE_DEFAULTS", False),
        )


Forwarder = Callable[
    [str, str, dict[str, str], bytes | None, list[tuple[str, str]] | None, list[tuple[str, tuple[str, Any, str]]] | None, DefaultProxySettings],
    Awaitable[httpx.Response],
]


def create_app(
    settings: DefaultProxySettings | None = None,
    forwarder: Forwarder | None = None,
) -> FastAPI:
    app_settings = settings or DefaultProxySettings.from_env()
    app_forwarder = forwarder or forward_to_mineru
    app = FastAPI(title="MinerU Default Proxy", version="0.1.0")

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "mineru_api_base_url": app_settings.mineru_api_base_url,
            "default_backend": app_settings.default_backend,
            "default_server_url": app_settings.default_server_url,
            "force_defaults": app_settings.force_defaults,
        }

    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"])
    async def proxy(path: str, request: Request) -> Response:
        target_url = build_target_url(app_settings.mineru_api_base_url, path, str(request.url.query))
        headers = filter_headers(dict(request.headers))

        body: bytes | None = None
        data: list[tuple[str, str]] | None = None
        files: list[tuple[str, tuple[str, Any, str]]] | None = None

        if should_inject_defaults(path, request):
            data, files = await build_multipart_payload(request, app_settings)
        else:
            body = await request.body()

        try:
            upstream_response = await app_forwarder(
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

        return Response(
            content=upstream_response.content,
            status_code=upstream_response.status_code,
            headers=filter_headers(dict(upstream_response.headers)),
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
) -> tuple[list[tuple[str, str]], list[tuple[str, tuple[str, Any, str]]]]:
    form = await request.form()
    data: list[tuple[str, str]] = []
    files: list[tuple[str, tuple[str, Any, str]]] = []

    for key, value in form.multi_items():
        if isinstance(value, UploadFile):
            await value.seek(0)
            files.append(
                (
                    key,
                    (
                        value.filename or "upload",
                        value.file,
                        value.content_type or "application/octet-stream",
                    ),
                )
            )
        else:
            data.append((key, str(value)))

    return apply_default_fields(data, settings), files


def apply_default_fields(fields: list[tuple[str, str]], settings: DefaultProxySettings) -> list[tuple[str, str]]:
    result = list(fields)
    existing = {key for key, _ in result}

    if settings.force_defaults:
        result = [(key, value) for key, value in result if key not in {"backend", "server_url"}]
        existing = {key for key, _ in result}

    if "backend" not in existing:
        result.append(("backend", settings.default_backend))
    if "server_url" not in existing:
        result.append(("server_url", settings.default_server_url))
    return result


async def forward_to_mineru(
    method: str,
    target_url: str,
    headers: dict[str, str],
    body: bytes | None,
    data: list[tuple[str, str]] | None,
    files: list[tuple[str, tuple[str, Any, str]]] | None,
    settings: DefaultProxySettings,
) -> httpx.Response:
    async with httpx.AsyncClient(timeout=settings.request_timeout) as client:
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


app = create_app()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the MinerU default proxy.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run("mineru_adapter.default_proxy:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
