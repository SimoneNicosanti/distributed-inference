from typing import override

from redis import RedisError
from redis.asyncio import Redis

from redis_access.contract.redis_address import HashMapRedisAddress, KeyRedisAddress
from redis_access.contract.redis_value import RedisValue
from redis_access.redis_reader.abc.batch_redis_reader import BatchRedisReader
from redis_access.redis_reader.abc.hash_fields_redis_reader import HashFieldsRedisReader
from redis_access.redis_reader.abc.multi_value_redis_reader import MultiValueRedisReader
from redis_access.redis_reader.abc.redis_existence_checker import RedisExistenceChecker
from redis_access.redis_reader.abc.single_value_redis_reader import (
    RedisReadError,
    RedisValueNotFoundError,
    SingleValueRedisReader,
)


class HashMapRedisReader(
    SingleValueRedisReader[HashMapRedisAddress],
    BatchRedisReader[HashMapRedisAddress],
    MultiValueRedisReader[KeyRedisAddress],
    HashFieldsRedisReader,
    RedisExistenceChecker[HashMapRedisAddress],
):
    @override
    async def read_single(self, redis: Redis, addr: HashMapRedisAddress) -> RedisValue:

        try:
            value = await redis.hget(addr.hash_map_key, addr.field_key)
        except RedisError as e:
            raise RedisReadError(f"Could not read from Redis address {addr}") from e

        if value is None:
            raise RedisValueNotFoundError(f"No value found for address {addr}")

        return value

    @override
    async def read_many(
        self, redis: Redis, addresses: tuple[HashMapRedisAddress, ...]
    ) -> tuple[RedisValue | None, ...]:
        if not addresses:
            return ()

        groups: dict[str, list[tuple[int, str]]] = {}
        for index, addr in enumerate(addresses):
            groups.setdefault(addr.hash_map_key, []).append((index, addr.field_key))

        try:
            async with redis.pipeline(transaction=False) as pipeline:
                for hash_map_key, fields in groups.items():
                    pipeline.hmget(hash_map_key, [field for _, field in fields])
                grouped_values = await pipeline.execute()
        except RedisError as e:
            raise RedisReadError("Could not read Redis hash batch") from e

        result: list[RedisValue | None] = [None] * len(addresses)
        for fields, values in zip(groups.values(), grouped_values, strict=True):
            for (index, _), value in zip(fields, values, strict=True):
                result[index] = value

        return tuple(result)

    @override
    async def read_all(
        self, redis: Redis, addr: KeyRedisAddress
    ) -> tuple[RedisValue, ...]:
        try:
            values = await redis.hvals(addr.key)
        except RedisError as e:
            raise RedisReadError(f"Could not read Redis hash {addr.key}") from e

        return tuple(values)

    @override
    async def read_fields(
        self, redis: Redis, addr: KeyRedisAddress
    ) -> tuple[RedisValue, ...]:
        try:
            fields = await redis.hkeys(addr.key)
        except RedisError as e:
            raise RedisReadError(f"Could not read Redis hash fields {addr.key}") from e

        return tuple(fields)

    @override
    async def exists(self, redis: Redis, addr: HashMapRedisAddress) -> bool:
        try:
            return bool(await redis.hexists(addr.hash_map_key, addr.field_key))
        except RedisError as e:
            raise RedisReadError(
                f"Could not check existence of Redis address {addr}"
            ) from e
