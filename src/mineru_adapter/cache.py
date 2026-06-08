from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Awaitable, Callable


UpstreamFactory = Callable[[], Awaitable[tuple[dict[str, Any], float]]]


@dataclass(slots=True)
class CacheEntry:
    expires_at: float
    response: dict[str, Any]


class UpstreamResponseCache:
    def __init__(self, max_entries: int, ttl_seconds: float, coalesce_requests: bool = True) -> None:
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self.coalesce_requests = coalesce_requests
        self._entries: OrderedDict[str, CacheEntry] = OrderedDict()
        self._inflight: dict[str, asyncio.Task[tuple[dict[str, Any], float]]] = {}
        self._lock = asyncio.Lock()

    @property
    def enabled(self) -> bool:
        return self.max_entries > 0 and self.ttl_seconds > 0

    async def get_or_call(self, key: str, factory: UpstreamFactory) -> tuple[dict[str, Any], float, str]:
        if not self.enabled:
            response, elapsed = await factory()
            return response, elapsed, "disabled"

        task: asyncio.Task[tuple[dict[str, Any], float]] | None = None
        owns_task = False

        async with self._lock:
            cached = self._get_valid_entry(key)
            if cached is not None:
                return cached, 0.0, "hit"

            if self.coalesce_requests:
                task = self._inflight.get(key)
                if task is not None:
                    cache_status = "coalesced"
                else:
                    task = asyncio.create_task(factory())
                    self._inflight[key] = task
                    owns_task = True
                    cache_status = "miss"
            else:
                task = asyncio.create_task(factory())
                owns_task = True
                cache_status = "miss"

        try:
            response, elapsed = await task
        except Exception:
            if owns_task:
                await self._remove_inflight(key, task)
            raise

        if owns_task:
            await self._store(key, response)
            await self._remove_inflight(key, task)
            return response, elapsed, cache_status
        return response, 0.0, cache_status

    def _get_valid_entry(self, key: str) -> dict[str, Any] | None:
        entry = self._entries.get(key)
        if entry is None:
            return None
        if entry.expires_at <= time.monotonic():
            self._entries.pop(key, None)
            return None
        self._entries.move_to_end(key)
        return entry.response

    async def _store(self, key: str, response: dict[str, Any]) -> None:
        async with self._lock:
            self._entries[key] = CacheEntry(
                expires_at=time.monotonic() + self.ttl_seconds,
                response=response,
            )
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)

    async def _remove_inflight(self, key: str, task: asyncio.Task[tuple[dict[str, Any], float]]) -> None:
        async with self._lock:
            if self._inflight.get(key) is task:
                self._inflight.pop(key, None)


def payload_cache_key(payload: dict[str, Any]) -> str:
    serialized = json.dumps(_cache_key_safe_value(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _cache_key_safe_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _cache_key_safe_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cache_key_safe_value(item) for item in value]
    if isinstance(value, str) and value.startswith("data:image/"):
        return _data_url_fingerprint(value)
    return value


def _data_url_fingerprint(value: str) -> dict[str, Any]:
    prefix, separator, _ = value.partition(",")
    return {
        "__type": "data-url-sha256",
        "prefix": prefix if separator else "data:image",
        "length": len(value),
        "sha256": hashlib.sha256(value.encode("utf-8")).hexdigest(),
    }
