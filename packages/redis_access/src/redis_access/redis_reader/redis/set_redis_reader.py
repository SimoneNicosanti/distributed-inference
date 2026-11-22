from typing import override

from redis import RedisError
from redis.asyncio import Redis

from redis_access.contract.redis_address import KeyRedisAddress
from redis_access.contract.redis_value import RedisValue
from redis_access.redis_reader.abc.multi_value_redis_reader import MultiValueRedisReader
from redis_access.redis_reader.abc.single_value_redis_reader import RedisReadError


class SetRedisReader(MultiValueRedisReader[KeyRedisAddress]):
    @override
    async def read_all(
        self, redis: Redis, addr: KeyRedisAddress
    ) -> tuple[RedisValue, ...]:
        try:
            values = [value async for value in redis.sscan_iter(addr.key)]
        except RedisError as e:
            raise RedisReadError(f"Could not scan Redis set {addr.key}") from e

        return tuple(values)
