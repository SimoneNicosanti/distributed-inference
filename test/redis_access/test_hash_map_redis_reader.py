from unittest.mock import MagicMock

import pytest
from fakeredis.aioredis import FakeRedis
from redis.asyncio import Redis
from redis.exceptions import ResponseError

from redis_access.contract.redis_address import HashMapRedisAddress, KeyRedisAddress
from redis_access.redis_reader.abc.single_value_redis_reader import RedisReadError
from redis_access.redis_reader.redis.hash_map_redis_reader import HashMapRedisReader


@pytest.mark.asyncio
@pytest.mark.parametrize("decode_responses", [False, True])
async def test_batch_preserves_order_duplicates_and_missing_values(
    decode_responses: bool,
) -> None:
    async with FakeRedis(decode_responses=decode_responses) as redis:
        await redis.hset("first", mapping={"shared": "one", "empty": ""})
        await redis.hset("second", mapping={"shared": "two"})

        values = await HashMapRedisReader().read_many(
            redis,
            (
                HashMapRedisAddress("second", "shared"),
                HashMapRedisAddress("first", "missing"),
                HashMapRedisAddress("first", "shared"),
                HashMapRedisAddress("second", "shared"),
                HashMapRedisAddress("first", "empty"),
                HashMapRedisAddress("missing", "shared"),
            ),
        )

    expected = (
        ("two", None, "one", "two", "", None)
        if decode_responses
        else (b"two", None, b"one", b"two", b"", None)
    )
    assert values == expected


@pytest.mark.asyncio
async def test_empty_batch_does_not_access_redis() -> None:
    redis = MagicMock(spec=Redis)

    assert await HashMapRedisReader().read_many(redis, ()) == ()

    redis.pipeline.assert_not_called()


@pytest.mark.asyncio
async def test_batch_wraps_redis_errors_without_returning_partial_results() -> None:
    async with FakeRedis() as redis:
        await redis.hset("valid", mapping={"field": "value"})
        await redis.set("invalid", "not a hash")

        with pytest.raises(RedisReadError) as caught:
            await HashMapRedisReader().read_many(
                redis,
                (
                    HashMapRedisAddress("valid", "field"),
                    HashMapRedisAddress("invalid", "field"),
                ),
            )

    assert isinstance(caught.value.__cause__, ResponseError)


@pytest.mark.asyncio
@pytest.mark.parametrize("decode_responses", [False, True])
async def test_read_all_preserves_duplicate_and_empty_values(
    decode_responses: bool,
) -> None:
    async with FakeRedis(decode_responses=decode_responses) as redis:
        await redis.hset(
            "catalog", mapping={"first": "same", "second": "same", "third": ""}
        )

        values = await HashMapRedisReader().read_all(redis, KeyRedisAddress("catalog"))

    assert isinstance(values, tuple)
    assert sorted(values) == (
        ["", "same", "same"] if decode_responses else [b"", b"same", b"same"]
    )


@pytest.mark.asyncio
async def test_read_all_returns_empty_tuple_for_missing_hash() -> None:
    async with FakeRedis() as redis:
        values = await HashMapRedisReader().read_all(redis, KeyRedisAddress("missing"))

    assert values == ()


@pytest.mark.asyncio
async def test_read_all_wraps_redis_errors() -> None:
    async with FakeRedis() as redis:
        await redis.set("invalid", "not a hash")

        with pytest.raises(RedisReadError) as caught:
            await HashMapRedisReader().read_all(redis, KeyRedisAddress("invalid"))

    assert isinstance(caught.value.__cause__, ResponseError)
