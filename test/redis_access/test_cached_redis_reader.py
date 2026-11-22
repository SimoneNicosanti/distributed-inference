import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from redis.asyncio import Redis

import redis_access.redis_reader.cached_redis_reader as reader_module
from redis_access.contract.redis_address import (
    HashMapRedisAddress,
    KeyRedisAddress,
    RedisAddress,
)
from redis_access.redis_reader.abc.redis_reader import RedisReader
from redis_access.redis_reader.cached_redis_reader import CachedRedisReader


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "addr", [KeyRedisAddress("plan"), HashMapRedisAddress("plans", "worker")]
)
@pytest.mark.parametrize("value", ["", "payload"])
async def test_reuses_cached_values(addr: RedisAddress, value: str) -> None:
    source = AsyncMock(spec=RedisReader)
    source.read.return_value = value
    reader = CachedRedisReader[RedisAddress](source)
    redis = Redis()

    assert await reader.read(redis, addr) == value
    assert await reader.read(redis, addr) == value
    source.read.assert_awaited_once_with(redis, addr)


@pytest.mark.parametrize("ttl_name", ["add_ttl_s", "access_ttl_s"])
@pytest.mark.parametrize("ttl", [0, -1])
def test_rejects_nonpositive_ttls(ttl_name: str, ttl: float) -> None:
    with pytest.raises(ValueError, match=r"Invalid .* ttl value"):
        CachedRedisReader(AsyncMock(spec=RedisReader), **{ttl_name: ttl})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("add_ttl", "access_ttl", "times"),
    [
        (10, None, (9, 10)),
        (None, 5, (4, 8, 13)),
        (10, 5, (4, 8, 10)),
    ],
)
async def test_ttls_expire_at_the_boundary(
    monkeypatch: pytest.MonkeyPatch,
    add_ttl: float | None,
    access_ttl: float | None,
    times: tuple[float, ...],
) -> None:
    now = 0.0
    monkeypatch.setattr(reader_module, "time", SimpleNamespace(monotonic=lambda: now))
    source = AsyncMock(spec=RedisReader)
    source.read.side_effect = ["first", "refreshed"]
    reader = CachedRedisReader(source, add_ttl_s=add_ttl, access_ttl_s=access_ttl)
    redis = Redis()
    addr = KeyRedisAddress("plan")

    assert await reader.read(redis, addr) == "first"
    for timestamp in times[:-1]:
        now = timestamp
        assert await reader.read(redis, addr) == "first"
    now = times[-1]
    assert await reader.read(redis, addr) == "refreshed"
    assert source.read.await_count == 2


@pytest.mark.asyncio
async def test_without_ttl_values_do_not_expire(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = 0.0
    monkeypatch.setattr(reader_module, "time", SimpleNamespace(monotonic=lambda: now))
    source = AsyncMock(spec=RedisReader)
    source.read.return_value = "plan"
    reader = CachedRedisReader(source)
    redis = Redis()
    addr = KeyRedisAddress("plan")

    assert await reader.read(redis, addr) == "plan"
    now = 1_000_000
    assert await reader.read(redis, addr) == "plan"
    source.read.assert_awaited_once()


@pytest.mark.asyncio
async def test_concurrent_misses_for_one_address_load_once() -> None:
    started, release = asyncio.Event(), asyncio.Event()

    async def load(redis: Redis, addr: RedisAddress) -> str:
        started.set()
        await release.wait()
        return "plan"

    source = AsyncMock(spec=RedisReader)
    source.read.side_effect = load
    reader = CachedRedisReader(source)
    redis = Redis()
    addr = KeyRedisAddress("plan")
    tasks = [asyncio.create_task(reader.read(redis, addr)) for _ in range(20)]
    try:
        await asyncio.wait_for(started.wait(), timeout=1)
        await asyncio.sleep(0)
        source.read.assert_awaited_once()
    finally:
        release.set()
        results = await asyncio.gather(*tasks)

    assert results == ["plan"] * 20
    source.read.assert_awaited_once()
    assert not reader._pending_reads


@pytest.mark.asyncio
async def test_different_addresses_load_concurrently() -> None:
    started, release = asyncio.Event(), asyncio.Event()
    calls = 0

    async def load(redis: Redis, addr: RedisAddress) -> str:
        nonlocal calls
        calls += 1
        if calls == 2:
            started.set()
        await release.wait()
        return "plan"

    source = AsyncMock(spec=RedisReader)
    source.read.side_effect = load
    reader = CachedRedisReader(source)
    redis = Redis()
    tasks = [
        asyncio.create_task(reader.read(redis, KeyRedisAddress(key)))
        for key in ("first", "second")
    ]
    try:
        await asyncio.wait_for(started.wait(), timeout=1)
        assert source.read.await_count == 2
    finally:
        release.set()
        await asyncio.gather(*tasks)


@pytest.mark.asyncio
async def test_cache_is_keyed_only_by_address() -> None:
    source = AsyncMock(spec=RedisReader)
    source.read.return_value = "plan"
    reader = CachedRedisReader(source)
    first, second = Redis(), Redis()
    addr = KeyRedisAddress("plan")

    assert await reader.read(first, addr) == "plan"
    assert await reader.read(second, addr) == "plan"
    assert await reader.read(first, addr) == "plan"
    source.read.assert_awaited_once_with(first, addr)


@pytest.mark.asyncio
async def test_failed_load_is_not_cached_and_can_be_retried() -> None:
    source = AsyncMock(spec=RedisReader)
    source.read.side_effect = [RuntimeError("unavailable"), "plan"]
    reader = CachedRedisReader(source)
    redis = Redis()
    addr = KeyRedisAddress("plan")

    with pytest.raises(RuntimeError, match="unavailable"):
        await reader.read(redis, addr)
    assert await reader.read(redis, addr) == "plan"
    assert await reader.read(redis, addr) == "plan"
    assert source.read.await_count == 2
    assert not reader._pending_reads


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_loader", [False, True])
async def test_cancellation_does_not_block_remaining_readers(
    cancel_loader: bool,
) -> None:
    started, release = asyncio.Event(), asyncio.Event()
    source = AsyncMock(spec=RedisReader)

    async def load(redis: Redis, addr: RedisAddress) -> str:
        if source.read.await_count == 1:
            started.set()
            await release.wait()
        return "plan"

    source.read.side_effect = load
    reader = CachedRedisReader(source)
    redis = Redis()
    addr = KeyRedisAddress("plan")
    loader = asyncio.create_task(reader.read(redis, addr))
    await asyncio.wait_for(started.wait(), timeout=1)
    waiter = asyncio.create_task(reader.read(redis, addr))
    await asyncio.sleep(0)
    cancelled, remaining = (loader, waiter) if cancel_loader else (waiter, loader)
    cancelled.cancel()
    try:
        with pytest.raises(asyncio.CancelledError):
            await cancelled
    finally:
        release.set()

    assert await asyncio.wait_for(remaining, timeout=1) == "plan"
    assert await reader.read(redis, addr) == "plan"
    assert source.read.await_count == (2 if cancel_loader else 1)
    assert not reader._pending_reads
