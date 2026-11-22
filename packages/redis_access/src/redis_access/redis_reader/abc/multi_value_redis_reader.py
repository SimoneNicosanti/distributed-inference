from abc import ABC, abstractmethod

from redis import asyncio as asyncio_redis

from redis_access.contract.redis_address import RedisAddress
from redis_access.contract.redis_value import RedisValue


class MultiValueRedisReader[AddrT: RedisAddress](ABC):
    @abstractmethod
    async def read_all(
        self, redis: asyncio_redis.Redis, addr: AddrT
    ) -> tuple[RedisValue, ...]: ...
