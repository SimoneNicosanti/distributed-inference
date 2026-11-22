import asyncio
import time
from dataclasses import dataclass
from typing import override

from redis.asyncio import Redis

from redis_access.contract.redis_address import (
    RedisAddress,
)
from redis_access.contract.redis_value import RedisValue
from redis_access.redis_reader.abc.single_value_redis_reader import (
    SingleValueRedisReader,
)


class CachedSingleValueRedisReader[AddrT: RedisAddress](SingleValueRedisReader[AddrT]):
    @dataclass(frozen=True, slots=True)
    class _CacheEntry:
        value: RedisValue
        add_time_s: float
        access_time_s: float

    @dataclass(slots=True)
    class _PendingRead:
        lock: asyncio.Lock
        users: int = 0

    def __init__(
        self,
        redis_reader: SingleValueRedisReader[AddrT],
        add_ttl_s: float | None = None,
        access_ttl_s: float | None = None,
    ) -> None:
        super().__init__()
        if add_ttl_s is not None and add_ttl_s <= 0:
            raise ValueError("Invalid add ttl value")
        if access_ttl_s is not None and access_ttl_s <= 0:
            raise ValueError("Invalid access ttl value")

        self._redis_reader = redis_reader
        self._add_ttl_s = add_ttl_s
        self._access_ttl_s = access_ttl_s

        self._cache: dict[AddrT, CachedSingleValueRedisReader._CacheEntry] = {}
        self._pending_reads: dict[AddrT, CachedSingleValueRedisReader._PendingRead] = {}

    @override
    async def read_single(self, redis: Redis, addr: AddrT) -> RedisValue:
        value = self._get_cached_value(addr)
        if value is not None:
            return value

        # Bookkeeping has no await: it is atomic within the reader's event loop.
        pending = self._pending_reads.get(addr)
        if pending is None:
            pending = self._PendingRead(asyncio.Lock())
            self._pending_reads[addr] = pending
        pending.users += 1

        try:
            async with pending.lock:
                value = self._get_cached_value(addr)
                if value is not None:
                    return value

                value = await self._redis_reader.read_single(redis, addr)
                now = time.monotonic()
                self._cache[addr] = self._CacheEntry(value, now, now)
                return value
        finally:
            pending.users -= 1
            if pending.users == 0:
                del self._pending_reads[addr]

    def _get_cached_value(self, addr: AddrT) -> RedisValue | None:
        entry = self._cache.get(addr)
        if entry is None:
            return None

        now = time.monotonic()
        if (
            self._add_ttl_s is not None and now - entry.add_time_s >= self._add_ttl_s
        ) or (
            self._access_ttl_s is not None
            and now - entry.access_time_s >= self._access_ttl_s
        ):
            del self._cache[addr]
            return None

        self._cache[addr] = self._CacheEntry(entry.value, entry.add_time_s, now)
        return entry.value
