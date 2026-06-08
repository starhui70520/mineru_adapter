from __future__ import annotations

import asyncio

from mineru_adapter.cache import UpstreamResponseCache, payload_cache_key


def test_payload_cache_key_is_stable_for_equivalent_payloads() -> None:
    left = {"model": "vl-model", "messages": [{"role": "user", "content": "x"}], "temperature": 0}
    right = {"temperature": 0, "messages": [{"content": "x", "role": "user"}], "model": "vl-model"}

    assert payload_cache_key(left) == payload_cache_key(right)


def test_cache_hit_returns_deep_copy() -> None:
    async def scenario() -> None:
        calls = 0
        cache = UpstreamResponseCache(max_entries=2, ttl_seconds=60)

        async def factory():
            nonlocal calls
            calls += 1
            return {"choices": [{"message": {"content": "ok"}}]}, 1.23

        first, first_elapsed, first_status = await cache.get_or_call("key", factory)
        first["choices"][0]["message"]["content"] = "mutated"
        second, second_elapsed, second_status = await cache.get_or_call("key", factory)

        assert calls == 1
        assert first_elapsed == 1.23
        assert first_status == "miss"
        assert second_elapsed == 0.0
        assert second_status == "hit"
        assert second["choices"][0]["message"]["content"] == "ok"

    asyncio.run(scenario())


def test_cache_coalesces_concurrent_requests() -> None:
    async def scenario() -> None:
        calls = 0
        cache = UpstreamResponseCache(max_entries=2, ttl_seconds=60, coalesce_requests=True)

        async def factory():
            nonlocal calls
            calls += 1
            await asyncio.sleep(0.01)
            return {"choices": [{"message": {"content": "ok"}}]}, 1.0

        results = await asyncio.gather(
            cache.get_or_call("same", factory),
            cache.get_or_call("same", factory),
            cache.get_or_call("same", factory),
        )

        assert calls == 1
        assert sorted(status for _, _, status in results) == ["coalesced", "coalesced", "miss"]

    asyncio.run(scenario())
