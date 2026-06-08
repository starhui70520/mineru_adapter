from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_UPSTREAM_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_UPSTREAM_MODEL = "vl-model"
DEFAULT_REQUEST_TIMEOUT = 120.0


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(slots=True)
class Settings:
    upstream_base_url: str = DEFAULT_UPSTREAM_BASE_URL
    upstream_model: str = DEFAULT_UPSTREAM_MODEL
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT
    drop_unsupported_params: bool = True
    strip_reasoning: bool = True
    disable_upstream_thinking: bool = True
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
            debug_dir=Path(debug_dir) if debug_dir else None,
        )
