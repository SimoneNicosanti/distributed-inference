import asyncio
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from fakeredis.aioredis import FakeRedis
from redis.exceptions import RedisError

from integration.shared.id_json import model_variant_id_to_json
from shared.model.model import ModelId
from shared.model.model_variant import ModelVariantId
from shared.service.service import ServiceId
from worker.adapters.outbound.profiling.reader.redis_pending_model_variant_consumer import (
    RedisPendingModelVariantConsumer,
)


@pytest_asyncio.fixture
async def redis() -> AsyncIterator[FakeRedis]:
    async with FakeRedis() as client:
        yield client


def variant(name: str) -> ModelVariantId:
    return ModelVariantId(ModelId(model_name=name), "fp32")


async def publish(redis: FakeRedis, model_variant_id: ModelVariantId) -> str | bytes:
    return await redis.xadd(
        "available-model-variants",
        {"model_variant_id": model_variant_id_to_json(model_variant_id)},
    )


async def read_round(
    consumer: RedisPendingModelVariantConsumer,
) -> list[ModelVariantId]:
    async with asyncio.timeout(1):
        return [model_id async for model_id in consumer.pending_model_variant_ids()]


@pytest.mark.asyncio
async def test_round_yields_every_unacknowledged_message_once(redis: FakeRedis) -> None:
    first, second = variant("first"), variant("second")
    first_id = await publish(redis, first)
    second_id = await publish(redis, second)
    consumer = RedisPendingModelVariantConsumer(ServiceId(id="worker"), redis)

    assert await read_round(consumer) == [first, second]
    assert consumer._message_ids_map == {first: {first_id}, second: {second_id}}
    assert await read_round(consumer) == [first, second]

    await consumer.acknowledge(second)
    assert second not in consumer._message_ids_map
    assert await read_round(consumer) == [first]


@pytest.mark.asyncio
async def test_arrivals_during_profiling_wait_for_the_next_round(
    redis: FakeRedis,
) -> None:
    first, second, third = variant("first"), variant("second"), variant("third")
    await publish(redis, first)
    await publish(redis, second)
    consumer = RedisPendingModelVariantConsumer(ServiceId(id="worker"), redis)
    batch = consumer.pending_model_variant_ids()

    assert await anext(batch) == first
    await publish(redis, third)
    assert [model_id async for model_id in batch] == [second]
    assert await read_round(consumer) == [first, second, third]


@pytest.mark.asyncio
async def test_duplicate_variant_messages_keep_all_entry_ids(redis: FakeRedis) -> None:
    model = variant("duplicate")
    entry_ids = {await publish(redis, model), await publish(redis, model)}
    consumer = RedisPendingModelVariantConsumer(ServiceId(id="worker"), redis)

    assert await read_round(consumer) == [model, model]
    assert consumer._message_ids_map[model] == entry_ids
    await consumer.acknowledge(model)

    assert consumer._message_ids_map == {}
    assert await read_round(consumer) == []
    pending = await redis.xpending("available-model-variants", "profiling:worker")
    assert pending["pending"] == 0


@pytest.mark.asyncio
async def test_restart_recovers_failures_and_reads_new_messages(
    redis: FakeRedis,
) -> None:
    first, second, third = variant("first"), variant("second"), variant("third")
    await publish(redis, first)
    await publish(redis, second)
    worker_id = ServiceId(id="worker")
    consumer = RedisPendingModelVariantConsumer(worker_id, redis)
    assert await read_round(consumer) == [first, second]
    await consumer.acknowledge_fail(first)
    await consumer.acknowledge(second)
    await publish(redis, third)

    restarted = RedisPendingModelVariantConsumer(worker_id, redis)
    assert await read_round(restarted) == [first, third]


@pytest.mark.asyncio
async def test_each_worker_receives_the_same_available_variants(
    redis: FakeRedis,
) -> None:
    model = variant("shared")
    await publish(redis, model)
    first = RedisPendingModelVariantConsumer(ServiceId(id="worker-1"), redis)
    second = RedisPendingModelVariantConsumer(ServiceId(id="worker-2"), redis)

    assert await read_round(first) == [model]
    await first.acknowledge(model)
    assert await read_round(first) == []
    assert await read_round(second) == [model]


@pytest.mark.asyncio
async def test_failed_ack_keeps_the_mapping(
    redis: FakeRedis, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = variant("retry")
    entry_id = await publish(redis, model)
    consumer = RedisPendingModelVariantConsumer(ServiceId(id="worker"), redis)
    assert await read_round(consumer) == [model]
    monkeypatch.setattr(redis, "xack", AsyncMock(side_effect=RedisError("unavailable")))

    with pytest.raises(RedisError, match="unavailable"):
        await consumer.acknowledge(model)

    assert consumer._message_ids_map[model] == {entry_id}
