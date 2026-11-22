from abc import ABC, abstractmethod

from redis import asyncio as asyncio_redis

from redis_access.contract.redis_address import RedisAddress


class RedisExistenceChecker[AddrT: RedisAddress](ABC):
    @abstractmethod
    async def exists(self, redis: asyncio_redis.Redis, addr: AddrT) -> bool: ...
