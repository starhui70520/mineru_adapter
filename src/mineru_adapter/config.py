from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_UPSTREAM_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_UPSTREAM_MODEL = "vl-model"
DEFAULT_REQUEST_TIMEOUT = 120.0
DEFAULT_LAYOUT_MAX_TOKENS = 1024
DEFAULT_TEXT_MAX_TOKENS = 2048
DEFAULT_TABLE_MAX_TOKENS = 2048
DEFAULT_FORMULA_MAX_TOKENS = 512
DEFAULT_IMAGE_MAX_TOKENS = 1024
DEFAULT_LAYOUT_MAX_IMAGE_SIDE = 896
DEFAULT_LAYOUT_JPEG_QUALITY = 90
DEFAULT_ADAPTER_CACHE_SIZE = 256
DEFAULT_ADAPTER_CACHE_TTL_SECONDS = 3600.0


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


@dataclass(slots=True)
class Settings:
    upstream_base_url: str = DEFAULT_UPSTREAM_BASE_URL
    upstream_model: str = DEFAULT_UPSTREAM_MODEL
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT
    drop_unsupported_params: bool = True
    strip_reasoning: bool = True
    disable_upstream_thinking: bool = True
    layout_max_tokens: int = DEFAULT_LAYOUT_MAX_TOKENS
    text_max_tokens: int = DEFAULT_TEXT_MAX_TOKENS
    table_max_tokens: int = DEFAULT_TABLE_MAX_TOKENS
    formula_max_tokens: int = DEFAULT_FORMULA_MAX_TOKENS
    image_max_tokens: int = DEFAULT_IMAGE_MAX_TOKENS
    layout_max_image_side: int = DEFAULT_LAYOUT_MAX_IMAGE_SIDE
    layout_jpeg_quality: int = DEFAULT_LAYOUT_JPEG_QUALITY
    adapter_cache_size: int = DEFAULT_ADAPTER_CACHE_SIZE
    adapter_cache_ttl_seconds: float = DEFAULT_ADAPTER_CACHE_TTL_SECONDS
    adapter_coalesce_requests: bool = True
    debug_async: bool = True
    debug_dir: Path | None = None

    @classmethod
    def from_env(cls) -> "Settings":
        debug_dir = os.getenv("ADAPTER_DEBUG_DIR")
        return cls(
            upstream_base_url=os.getenv("UPSTREAM_BASE_URL", DEFAULT_UPSTREAM_BASE_URL).rstrip("/"),
            upstream_model=os.getenv("UPSTREAM_MODEL", DEFAULT_UPSTREAM_MODEL),
            request_timeout=float(os.getenv("REQUEST_TIMEOUT", str(DEFAULT_REQUEST_TIMEOUT))),
            drop_unsupported_params=_bool_env("DROP_UNSUPPORTED_PARAMS", True),
            strip_reasoning=_bool_env("STRIP_REASONING", True),
            disable_upstream_thinking=_bool_env("DISABLE_UPSTREAM_THINKING", True),
            layout_max_tokens=_int_env("LAYOUT_MAX_TOKENS", DEFAULT_LAYOUT_MAX_TOKENS),
            text_max_tokens=_int_env("TEXT_MAX_TOKENS", DEFAULT_TEXT_MAX_TOKENS),
            table_max_tokens=_int_env("TABLE_MAX_TOKENS", DEFAULT_TABLE_MAX_TOKENS),
            formula_max_tokens=_int_env("FORMULA_MAX_TOKENS", DEFAULT_FORMULA_MAX_TOKENS),
            image_max_tokens=_int_env("IMAGE_MAX_TOKENS", DEFAULT_IMAGE_MAX_TOKENS),
            layout_max_image_side=_int_env("LAYOUT_MAX_IMAGE_SIDE", DEFAULT_LAYOUT_MAX_IMAGE_SIDE),
            layout_jpeg_quality=_int_env("LAYOUT_JPEG_QUALITY", DEFAULT_LAYOUT_JPEG_QUALITY),
            adapter_cache_size=_int_env("ADAPTER_CACHE_SIZE", DEFAULT_ADAPTER_CACHE_SIZE),
            adapter_cache_ttl_seconds=float(os.getenv("ADAPTER_CACHE_TTL_SECONDS", str(DEFAULT_ADAPTER_CACHE_TTL_SECONDS))),
            adapter_coalesce_requests=_bool_env("ADAPTER_COALESCE_REQUESTS", True),
            debug_async=_bool_env("ADAPTER_DEBUG_ASYNC", True),
            debug_dir=Path(debug_dir) if debug_dir else None,
        )
