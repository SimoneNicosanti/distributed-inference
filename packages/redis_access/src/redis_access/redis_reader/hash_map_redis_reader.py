from typing import override

from redis import RedisError
from redis.asyncio import Redis

from redis_access.contract.redis_address import HashMapRedisAddress
from redis_access.contract.redis_value import RedisValue
from redis_access.redis_reader.abc.redis_reader import (
    RedisReader,
    RedisReadError,
    RedisValueNotFoundError,
)


class HashMapRedisReader(RedisReader[HashMapRedisAddress]):
    @override
    async def read(self, redis: Redis, addr: HashMapRedisAddress) -> RedisValue:

        try:
            value = await redis.hget(addr.hash_map_key, addr.field_key)
        except RedisError as e:
            raise RedisReadError(f"Could not read from Redis address {addr}") from e

        if value is None:
            raise RedisValueNotFoundError(f"No value found for address {addr}")

        return value
