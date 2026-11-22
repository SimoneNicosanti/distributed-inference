from unittest.mock import AsyncMock

import pytest
from fakeredis.aioredis import FakeRedis
from redis.asyncio import Redis
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import ResponseError

from redis_access.contract.redis_address import HashMapRedisAddress, KeyRedisAddress
from redis_access.redis_reader.abc.single_value_redis_reader import RedisReadError
from redis_access.redis_reader.redis.hash_map_redis_reader import HashMapRedisReader
from redis_access.redis_reader.redis.key_redis_reader import KeyRedisReader


@pytest.mark.asyncio
@pytest.mark.parametrize("decode_responses", [False, True])
async def test_read_fields_returns_raw_field_names_and_handles_missing_hash(
    decode_responses: bool,
) -> None:
    async with FakeRedis(decode_responses=decode_responses) as redis:
        await redis.hset("profiles", mapping={"": "first", "variante-à": "second"})

        fields = await HashMapRedisReader().read_fields(
            redis, KeyRedisAddress("profiles")
        )
        missing = await HashMapRedisReader().read_fields(
            redis, KeyRedisAddress("missing")
        )

    assert isinstance(fields, tuple)
    expected = {"", "variante-à"}
    assert set(fields) == (
        expected if decode_responses else {field.encode() for field in expected}
    )
    assert len(fields) == 2
    assert missing == ()


@pytest.mark.asyncio
async def test_hash_exists_distinguishes_fields_and_accepts_empty_values() -> None:
    reader = HashMapRedisReader()
    async with FakeRedis() as redis:
        await redis.hset("profiles", mapping={"existing": ""})

        assert (
            await reader.exists(redis, HashMapRedisAddress("profiles", "existing"))
            is True
        )
        assert (
            await reader.exists(redis, HashMapRedisAddress("profiles", "missing"))
            is False
        )
        assert (
            await reader.exists(redis, HashMapRedisAddress("missing", "existing"))
            is False
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["read_fields", "exists"])
async def test_hash_capabilities_wrap_wrong_type_errors(operation: str) -> None:
    reader = HashMapRedisReader()
    async with FakeRedis() as redis:
        await redis.set("invalid", "not a hash")

        with pytest.raises(RedisReadError) as caught:
            if operation == "read_fields":
                await reader.read_fields(redis, KeyRedisAddress("invalid"))
            else:
                await reader.exists(redis, HashMapRedisAddress("invalid", "field"))

    assert isinstance(caught.value.__cause__, ResponseError)


@pytest.mark.asyncio
@pytest.mark.parametrize("key_type", ["string", "hash"])
async def test_key_exists_checks_the_key_independently_of_its_value_type(
    key_type: str,
) -> None:
    reader = KeyRedisReader()
    async with FakeRedis() as redis:
        if key_type == "string":
            await redis.set("existing", "")
        else:
            await redis.hset("existing", mapping={"field": "value"})

        assert await reader.exists(redis, KeyRedisAddress("existing")) is True
        assert await reader.exists(redis, KeyRedisAddress("missing")) is False


@pytest.mark.asyncio
async def test_key_exists_wraps_connection_errors() -> None:
    redis = AsyncMock(spec=Redis)
    failure = RedisConnectionError("unavailable")
    redis.exists.side_effect = failure

    with pytest.raises(RedisReadError) as caught:
        await KeyRedisReader().exists(redis, KeyRedisAddress("model"))

    assert caught.value.__cause__ is failure
