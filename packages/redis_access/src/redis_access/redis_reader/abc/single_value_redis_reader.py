from abc import ABC, abstractmethod

from redis import asyncio as asyncio_redis

from redis_access.contract.redis_address import RedisAddress
from redis_access.contract.redis_value import RedisValue


class RedisReadError(Exception): ...


class RedisValueNotFoundError(RedisReadError): ...


class SingleValueRedisReader[AddrT: RedisAddress](ABC):
    @abstractmethod
    async def read_single(
        self, redis: asyncio_redis.Redis, addr: AddrT
    ) -> RedisValue: ...
