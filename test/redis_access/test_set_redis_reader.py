import pytest
from fakeredis.aioredis import FakeRedis
from redis.exceptions import ResponseError

from redis_access.contract.redis_address import KeyRedisAddress
from redis_access.redis_reader.abc.single_value_redis_reader import RedisReadError
from redis_access.redis_reader.redis.set_redis_reader import SetRedisReader


@pytest.mark.asyncio
@pytest.mark.parametrize("decode_responses", [False, True])
async def test_set_scan_returns_raw_members_and_handles_missing_sets(
    decode_responses: bool,
) -> None:
    expected = {"", "variante-à"} | {f"variant-{index}" for index in range(25)}
    async with FakeRedis(decode_responses=decode_responses) as redis:
        await redis.sadd("variants", *expected)

        members = await SetRedisReader().read_all(redis, KeyRedisAddress("variants"))
        missing = await SetRedisReader().read_all(redis, KeyRedisAddress("missing"))

    assert isinstance(members, tuple)
    assert set(members) == (
        expected if decode_responses else {value.encode() for value in expected}
    )
    assert len(members) == len(expected)
    assert missing == ()


@pytest.mark.asyncio
async def test_set_scan_wraps_redis_errors() -> None:
    async with FakeRedis() as redis:
        await redis.set("invalid", "not a set")

        with pytest.raises(RedisReadError) as caught:
            await SetRedisReader().read_all(redis, KeyRedisAddress("invalid"))

    assert isinstance(caught.value.__cause__, ResponseError)
