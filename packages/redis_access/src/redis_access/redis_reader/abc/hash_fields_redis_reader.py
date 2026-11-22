from abc import ABC, abstractmethod

from redis import asyncio as asyncio_redis

from redis_access.contract.redis_address import KeyRedisAddress
from redis_access.contract.redis_value import RedisValue


class HashFieldsRedisReader(ABC):
    @abstractmethod
    async def read_fields(
        self, redis: asyncio_redis.Redis, addr: KeyRedisAddress
    ) -> tuple[RedisValue, ...]:
        """Return hash fields in unspecified order; a missing hash returns ()."""
        ...
