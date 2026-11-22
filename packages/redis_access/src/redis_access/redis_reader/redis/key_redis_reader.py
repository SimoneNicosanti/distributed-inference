from typing import override

from redis import RedisError
from redis.asyncio import Redis

from redis_access.contract.redis_address import KeyRedisAddress
from redis_access.contract.redis_value import RedisValue
from redis_access.redis_reader.abc.redis_existence_checker import RedisExistenceChecker
from redis_access.redis_reader.abc.single_value_redis_reader import (
    RedisReadError,
    RedisValueNotFoundError,
    SingleValueRedisReader,
)


class KeyRedisReader(
    SingleValueRedisReader[KeyRedisAddress], RedisExistenceChecker[KeyRedisAddress]
):
    @override
    async def read_single(self, redis: Redis, addr: KeyRedisAddress) -> RedisValue:

        try:
            value = await redis.get(addr.key)
        except RedisError as e:
            raise RedisReadError(f"Could not read from Redis address {addr}") from e

        if value is None:
            raise RedisValueNotFoundError(f"No value found for address {addr}")

        return value

    @override
    async def exists(self, redis: Redis, addr: KeyRedisAddress) -> bool:
        try:
            return bool(await redis.exists(addr.key))
        except RedisError as e:
            raise RedisReadError(
                f"Could not check existence of Redis address {addr}"
            ) from e
