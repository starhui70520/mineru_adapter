from __future__ import annotations

import argparse
from contextlib import asynccontextmanager
from typing import Any, Awaitable, Callable

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from .config import Settings
from .proxy import build_upstream_payload, call_upstream, openai_models_response, rewrite_upstream_response, write_debug_record

UpstreamCaller = Callable[[dict[str, Any], Settings], Awaitable[tuple[dict[str, Any], float]]]


def create_app(settings: Settings | None = None, upstream_caller: UpstreamCaller = call_upstream) -> FastAPI:
    app_settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        async with httpx.AsyncClient(timeout=app_settings.request_timeout) as client:
            app.state.upstream_client = client
            yield

    app = FastAPI(title="MinerU Adapter", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "upstream_base_url": app_settings.upstream_base_url,
            "upstream_model": app_settings.upstream_model,
        }

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        return openai_models_response(app_settings)

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> JSONResponse:
        request_body = await request.json()
        if not isinstance(request_body, dict):
            raise HTTPException(status_code=400, detail="JSON body must be an object")

        outbound_payload: dict[str, Any] | None = None
        task = None
        image_size = None
        upstream_response = None
        rewritten_response = None
        raw_content = None
        elapsed_seconds = None
        parse_error = None

        try:
            outbound_payload, task, image_size = build_upstream_payload(request_body, app_settings)
            if upstream_caller is call_upstream:
                upstream_response, elapsed_seconds = await call_upstream(
                    outbound_payload,
                    app_settings,
                    client=getattr(request.app.state, "upstream_client", None),
                )
            else:
                upstream_response, elapsed_seconds = await upstream_caller(outbound_payload, app_settings)
            rewritten_response, raw_content, parse_error = rewrite_upstream_response(
                upstream_response,
                task,
                image_size=image_size,
                settings=app_settings,
            )
            write_debug_record(
                app_settings,
                request_body,
                outbound_payload,
                upstream_response,
                rewritten_response,
                task,
                elapsed_seconds,
                raw_content,
                parse_error,
            )
            return JSONResponse(rewritten_response)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except httpx.HTTPStatusError as exc:
            detail = {"upstream_status_code": exc.response.status_code, "upstream_response": exc.response.text}
            raise HTTPException(status_code=502, detail=detail) from exc
        except Exception as exc:
            if outbound_payload is not None and task is not None:
                write_debug_record(
                    app_settings,
                    request_body,
                    outbound_payload,
                    upstream_response,
                    rewritten_response,
                    task,
                    elapsed_seconds,
                    raw_content,
                    parse_error,
                    exception=repr(exc),
                )
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    return app


app = create_app()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the MinerU adapter.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18000)
    args = parser.parse_args()
    uvicorn.run("mineru_adapter.api:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
