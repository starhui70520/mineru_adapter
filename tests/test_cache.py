from __future__ import annotations

import asyncio
import hashlib

from mineru_adapter.cache import UpstreamResponseCache, _cache_key_safe_value, _sha256_text, payload_cache_key


def test_payload_cache_key_is_stable_for_equivalent_payloads() -> None:
    left = {"model": "vl-model", "messages": [{"role": "user", "content": "x"}], "temperature": 0}
    right = {"temperature": 0, "messages": [{"content": "x", "role": "user"}], "model": "vl-model"}

    assert payload_cache_key(left) == payload_cache_key(right)


def test_payload_cache_key_fingerprints_data_urls() -> None:
    left = {"messages": [{"content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]}]}
    right = {"messages": [{"content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,BBBB"}}]}]}

    assert payload_cache_key(left) != payload_cache_key(right)
    safe = _cache_key_safe_value(left)
    fingerprint = safe["messages"][0]["content"][0]["image_url"]["url"]
    assert fingerprint["__type"] == "data-url-sha256"
    assert fingerprint["prefix"] == "data:image/png;base64"
    assert fingerprint["length"] == len("data:image/png;base64,AAAA")
    assert "AAAA" not in str(safe)


def test_sha256_text_matches_single_encode_digest() -> None:
    value = "data:image/png;base64," + ("ABCD" * 10)

    assert _sha256_text(value, chunk_size=7) == hashlib.sha256(value.encode("utf-8")).hexdigest()


def test_cache_reuses_response_objects_without_copying() -> None:
    async def scenario() -> None:
        calls = 0
        cache = UpstreamResponseCache(max_entries=2, ttl_seconds=60)
        response = {"choices": [{"message": {"content": "ok"}}]}

        async def factory():
            nonlocal calls
            calls += 1
            return response, 1.23

        first, first_elapsed, first_status = await cache.get_or_call("key", factory)
        second, second_elapsed, second_status = await cache.get_or_call("key", factory)

        assert calls == 1
        assert first_elapsed == 1.23
        assert first_status == "miss"
        assert second_elapsed == 0.0
        assert second_status == "hit"
        assert first is response
        assert second is response

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
