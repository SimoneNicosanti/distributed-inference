from abc import ABC, abstractmethod

from redis import asyncio as asyncio_redis

from redis_access.contract.redis_address import RedisAddress
from redis_access.contract.redis_value import RedisValue


class BatchRedisReader[AddrT: RedisAddress](ABC):
    @abstractmethod
    async def read_many(
        self,
        redis: asyncio_redis.Redis,
        addresses: tuple[AddrT, ...],
    ) -> tuple[RedisValue | None, ...]: ...
