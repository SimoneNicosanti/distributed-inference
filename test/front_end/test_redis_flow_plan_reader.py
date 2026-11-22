from collections import Counter
from unittest.mock import AsyncMock

import pytest
from redis.asyncio import Redis
from redis.exceptions import ConnectionError as RedisConnectionError

from front_end.adapters.outbound.inference.redis_flow_plan_reader import (
    RedisFlowPlanReader,
)
from front_end.application.ports.outbound.inference.flow_plan_reader import (
    FlowPlanReadError,
)
from integration.dto.flow_plan.flow_plan_dto import FlowPlanDto, ModelSpecsDto
from redis_access.address_factory.front_end_plan.address_factory import (
    HashMapFlowPlanAddressFactory,
)
from redis_access.redis_reader.cached.cached_single_value_redis_reader import (
    CachedSingleValueRedisReader,
)
from redis_access.redis_reader.redis.hash_map_redis_reader import (
    HashMapRedisReader,
)
from shared.flow.flow import FlowId
from shared.model.model import ModelId
from shared.model.model_variant import ModelVariantId
from shared.plan.hop_logical_id import PlannedIngressReplicaId, PlannedSinkReplicaId
from shared.plan.plan_version import PlanVersion
from shared.service.service import ServiceId


def _redis() -> AsyncMock:
    client = AsyncMock(spec=Redis)
    client.hget = AsyncMock()
    return client


def _reader(redis: Redis) -> RedisFlowPlanReader:
    return RedisFlowPlanReader(
        redis=redis,
        current_version_reader=HashMapRedisReader(),
        versioned_plan_reader=CachedSingleValueRedisReader(
            HashMapRedisReader(), add_ttl_s=60.0
        ),
        address_factory=HashMapFlowPlanAddressFactory(),
    )


def _plan(version: int = 3) -> FlowPlanDto:
    variant = ModelVariantId(ModelId("model"), "fp32")
    return FlowPlanDto(
        plan_version=PlanVersion(version),
        model_specs=ModelSpecsDto(model_variant_id=variant, inputs=()),
        ingress_replica_id=PlannedIngressReplicaId(variant, 0),
        ingress_worker_id=ServiceId(id="ingress"),
        sink_replica_id=PlannedSinkReplicaId(variant, 1),
        sink_worker_id=ServiceId(id="sink"),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("value", ["3", b"3"])
async def test_reads_current_version_from_string_or_bytes(value: str | bytes) -> None:
    flow_id = FlowId()
    redis = _redis()
    redis.hget.return_value = value
    reader = _reader(redis)

    version = await reader.get_current_plan_version_for_flow(flow_id)

    assert version == PlanVersion(version_number=3)
    redis.hget.assert_awaited_once_with(
        f"flow-plans:{flow_id.flow_id}",
        "active-version",
    )


@pytest.mark.asyncio
async def test_missing_current_version_raises_read_error() -> None:
    redis = _redis()
    redis.hget.return_value = None
    reader = _reader(redis)

    with pytest.raises(FlowPlanReadError, match="Could not read current flow plan"):
        await reader.get_current_plan_version_for_flow(FlowId())


@pytest.mark.asyncio
@pytest.mark.parametrize("value", ["", "invalid", "1.5", "0", "-1"])
async def test_invalid_current_version_raises_read_error(value: str) -> None:
    redis = _redis()
    redis.hget.return_value = value
    reader = _reader(redis)

    with pytest.raises(FlowPlanReadError):
        await reader.get_current_plan_version_for_flow(FlowId())


@pytest.mark.asyncio
@pytest.mark.parametrize("as_bytes", [False, True])
async def test_reads_versioned_plan_from_string_or_bytes(as_bytes: bool) -> None:
    dto = _plan()
    payload = dto.model_dump_json()
    redis = _redis()
    redis.hget.return_value = payload.encode() if as_bytes else payload

    plan = await _reader(redis).get_flow_plan_by_plan_version(
        FlowId(), dto.plan_version
    )

    assert plan.plan_version == dto.plan_version
    assert plan.ingress_worker_id == dto.ingress_worker_id
    assert plan.sink_worker_id == dto.sink_worker_id


@pytest.mark.asyncio
@pytest.mark.parametrize("read_current_version", [False, True])
async def test_redis_errors_are_translated(read_current_version: bool) -> None:
    redis = _redis()
    redis.hget.side_effect = RedisConnectionError("unavailable")
    reader = _reader(redis)

    with pytest.raises(FlowPlanReadError):
        if read_current_version:
            await reader.get_current_plan_version_for_flow(FlowId())
        else:
            await reader.get_flow_plan_by_plan_version(FlowId(), PlanVersion(3))


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [None, "invalid", _plan(4).model_dump_json()])
async def test_missing_invalid_or_mismatched_plan_raises_read_error(
    value: str | None,
) -> None:
    redis = _redis()
    redis.hget.return_value = value

    with pytest.raises(FlowPlanReadError):
        await _reader(redis).get_flow_plan_by_plan_version(FlowId(), PlanVersion(3))


@pytest.mark.asyncio
async def test_current_version_is_fresh_while_versioned_plans_are_cached() -> None:
    redis = _redis()
    reader = _reader(redis)
    flow_id = FlowId()
    hash_key = f"flow-plans:{flow_id.flow_id}"
    active_field = "active-version"
    old, new = _plan(3), _plan(4)
    old_field = f"version-{old.plan_version.version_number}"
    new_field = f"version-{new.plan_version.version_number}"
    values = {active_field: "3", old_field: old.model_dump_json()}

    async def hget(key: str, field: str) -> str | None:
        assert key == hash_key
        return values.get(field)

    redis.hget.side_effect = hget
    assert await reader.get_current_plan_version_for_flow(flow_id) == old.plan_version
    for _ in range(2):
        plan = await reader.get_flow_plan_by_plan_version(flow_id, old.plan_version)
        assert plan.plan_version == old.plan_version

    values[active_field] = "4"
    values[new_field] = new.model_dump_json()
    assert await reader.get_current_plan_version_for_flow(flow_id) == new.plan_version
    for _ in range(2):
        plan = await reader.get_flow_plan_by_plan_version(flow_id, new.plan_version)
        assert plan.plan_version == new.plan_version
    assert (
        await reader.get_flow_plan_by_plan_version(flow_id, old.plan_version)
    ).plan_version == old.plan_version

    assert Counter(call.args for call in redis.hget.await_args_list) == {
        (hash_key, active_field): 2,
        (hash_key, old_field): 1,
        (hash_key, new_field): 1,
    }


@pytest.mark.asyncio
async def test_versioned_payload_cache_expires_after_sixty_seconds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    import redis_access.redis_reader.cached.cached_single_value_redis_reader as cache_module

    now = 0.0
    monkeypatch.setattr(cache_module, "time", SimpleNamespace(monotonic=lambda: now))
    old = _plan()
    updated = old.model_copy(update={"sink_worker_id": ServiceId(id="updated-sink")})
    redis = _redis()
    redis.hget.side_effect = [old.model_dump_json(), updated.model_dump_json()]
    reader = _reader(redis)
    flow_id = FlowId()
    assert (
        await reader.get_flow_plan_by_plan_version(flow_id, old.plan_version)
    ).sink_worker_id == old.sink_worker_id
    now = 59.9
    assert (
        await reader.get_flow_plan_by_plan_version(flow_id, old.plan_version)
    ).sink_worker_id == old.sink_worker_id
    now = 60.0
    assert (
        await reader.get_flow_plan_by_plan_version(flow_id, old.plan_version)
    ).sink_worker_id == updated.sink_worker_id
    assert redis.hget.await_count == 2
