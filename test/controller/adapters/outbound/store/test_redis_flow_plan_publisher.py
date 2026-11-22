from collections.abc import AsyncGenerator
from dataclasses import replace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from fakeredis.aioredis import FakeRedis
from redis import asyncio as asyncio_redis
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import ResponseError

from controller.adapters.outbound.plan_publishing.mappers.flow_plan_mapper import (
    FlowPlanMapper,
)
from controller.adapters.outbound.plan_publishing.mappers.worker_plan_mapper import (
    WorkerPlanMapper,
)
from controller.adapters.outbound.plan_publishing.redis_plan_publisher import (
    RedisPlanPublisher,
)
from controller.application.ports.outbound.plan_publishing.plan_publisher import (
    PlanPublisher,
)
from controller.domain.plan.deployment_plan import DeploymentPlan
from controller.domain.plan.failure_plan import FailurePlan
from controller.domain.plan.flow_plan import FlowPlan
from controller.domain.plan.hop_placement_plan import HopPlacementPlan
from controller.domain.plan.hop_routing_plan import HopRoutingPlan
from controller.domain.plan.partition_execution_plan import PartitionExecutionPlan
from controller.domain.plan.scheduling_plan import SchedulingPlan
from controller.domain.plan.worker_plan import WorkerPlan
from controller.domain.profile.model_variant_static_profile import TensorProperty
from front_end.adapters.outbound.inference.redis_flow_plan_reader import (
    RedisFlowPlanReader,
)
from integration.dto.flow_plan.flow_plan_dto import FlowPlanDto
from integration.dto.worker_plan.worker_plan_dto import WorkerPlanDto
from redis_access.address_factory.front_end_plan.address_factory import (
    HashMapFlowPlanAddressFactory,
)
from redis_access.address_factory.worker_plan.address_factory import (
    HashMapWorkerPlanAddressFactory,
)
from redis_access.redis_reader.cached_redis_reader import CachedRedisReader
from redis_access.redis_reader.hash_map_redis_reader import HashMapRedisReader
from shared.flow.flow import FlowId
from shared.model.model import ModelId
from shared.model.model_variant import ModelVariantId
from shared.plan.hop_logical_id import PlannedIngressReplicaId, PlannedSinkReplicaId
from shared.plan.plan_version import PlanVersion
from shared.service.service import ServiceId


def _plan(version: int = 3, variant_tag: str = "variant") -> FlowPlan:
    variant = ModelVariantId(ModelId("model"), variant_tag)
    return FlowPlan(
        plan_version=PlanVersion(version),
        model_variant_id=variant,
        input_tensor_properties={
            "scale": TensorProperty("float32", 4, ()),
            "features": TensorProperty("float32", 4, ("batch", 3)),
            "mask": TensorProperty("bool", 1, ("batch", 2)),
        },
        ingress_replica_id=PlannedIngressReplicaId(variant, 2),
        ingress_worker_id=ServiceId(),
        sink_replica_id=PlannedSinkReplicaId(variant, 5),
        sink_worker_id=ServiceId(),
    )


def _worker_plan(version: PlanVersion) -> WorkerPlan:
    return WorkerPlan(
        plan_version=version,
        deployment_plan=DeploymentPlan((), (), (), ()),
        scheduling_plan=SchedulingPlan((), ()),
        partition_execution_plan=PartitionExecutionPlan((), ()),
        hop_routing_plan=HopRoutingPlan((), ()),
        failure_plan=FailurePlan((), ()),
        hop_placement_plan=HopPlacementPlan({}),
    )


def _worker_plans(flows: dict[FlowId, FlowPlan]) -> dict[ServiceId, WorkerPlan]:
    return {
        worker: _worker_plan(plan.plan_version)
        for plan in flows.values()
        for worker in (plan.ingress_worker_id, plan.sink_worker_id)
    }


def _publisher(redis: asyncio_redis.Redis) -> RedisPlanPublisher:
    return RedisPlanPublisher(
        redis=redis,
        worker_plan_address_factory=HashMapWorkerPlanAddressFactory(),
        flow_plan_address_factory=HashMapFlowPlanAddressFactory(),
    )


@pytest_asyncio.fixture
async def redis() -> AsyncGenerator[FakeRedis]:
    async with FakeRedis(decode_responses=True) as connection:
        yield connection


def test_maps_model_contract_version_and_endpoint_identity() -> None:
    plan = _plan()
    dto = FlowPlanMapper.to_flow_plan_dto(plan)
    received = FlowPlanDto.model_validate_json(dto.model_dump_json())

    assert received.plan_version == plan.plan_version
    assert received.model_specs.model_variant_id == plan.model_variant_id
    assert received.ingress_replica_id == plan.ingress_replica_id
    assert received.ingress_worker_id == plan.ingress_worker_id
    assert received.sink_replica_id == plan.sink_replica_id
    assert received.sink_worker_id == plan.sink_worker_id
    assert {
        spec.name: (spec.shape, spec.dtype) for spec in received.model_specs.inputs
    } == {
        "features": (("batch", 3), "float32"),
        "mask": (("batch", 2), "bool"),
        "scale": ((), "float32"),
    }


@pytest.mark.asyncio
async def test_publishes_worker_and_flow_payloads_and_active_hash_fields(
    redis: FakeRedis,
) -> None:
    flows = {FlowId(): _plan(), FlowId(): _plan(variant_tag="other")}
    workers = _worker_plans(flows)
    publisher: PlanPublisher = _publisher(redis)

    await publisher.publish_plans(workers, flows)

    for worker_id, plan in workers.items():
        payload = await redis.hget(
            f"worker-plans:version-{plan.plan_version.version_number}",
            str(worker_id.id),
        )
        assert WorkerPlanDto.model_validate_json(payload) == (
            WorkerPlanMapper.to_worker_plan_dto(plan)
        )

    reader = RedisFlowPlanReader(
        redis,
        HashMapRedisReader(),
        CachedRedisReader(HashMapRedisReader(), add_ttl_s=60.0),
        HashMapFlowPlanAddressFactory(),
    )
    for flow_id, plan in flows.items():
        hash_key = f"flow-plans:{flow_id.flow_id}"
        active_field = "active-version"
        assert await redis.hget(hash_key, active_field) == "3"
        assert await redis.type(hash_key) == "hash"
        received = await reader.get_flow_plan_by_plan_version(
            flow_id, plan.plan_version
        )
        assert received.ingress_worker_id == plan.ingress_worker_id
        assert received.sink_worker_id == plan.sink_worker_id
        assert received.plan_version == plan.plan_version


@pytest.mark.asyncio
async def test_partial_update_preserves_other_active_flows_and_previous_plans(
    redis: FakeRedis,
) -> None:
    publisher = _publisher(redis)
    first, second = FlowId(), FlowId()
    old = {first: _plan(), second: _plan(variant_tag="other")}
    old_workers = _worker_plans(old)
    await publisher.publish_plans(old_workers, old)
    old_worker_payloads = await redis.hgetall("worker-plans:version-3")
    current = {first: replace(old[first], plan_version=PlanVersion(4))}

    await publisher.publish_plans(_worker_plans(current), current)

    active_field = "active-version"
    assert await redis.hget(f"flow-plans:{first.flow_id}", active_field) == "4"
    assert await redis.hget(f"flow-plans:{second.flow_id}", active_field) == "3"
    old_payload = await redis.hget(
        f"flow-plans:{first.flow_id}",
        f"version-{(PlanVersion(3)).version_number}",
    )
    assert FlowPlanDto.model_validate_json(old_payload) == (
        FlowPlanMapper.to_flow_plan_dto(old[first])
    )
    assert await redis.hgetall("worker-plans:version-3") == (old_worker_payloads)
    assert set(await redis.hkeys("worker-plans:version-4")) == {
        str(worker.id) for worker in _worker_plans(current)
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_key", ["worker", "flow"])
async def test_staging_error_keeps_old_active_version_even_after_partial_writes(
    redis: FakeRedis, invalid_key: str
) -> None:
    flow_id, other_id = FlowId(), FlowId()
    current = _plan(4)
    previous = replace(current, plan_version=PlanVersion(3))
    flows = {flow_id: current, other_id: _plan(4, "other")}
    hash_key = f"flow-plans:{flow_id.flow_id}"
    active_field = "active-version"
    await redis.hset(
        hash_key,
        mapping={
            active_field: "3",
            f"version-{previous.plan_version.version_number}": (
                FlowPlanMapper.to_flow_plan_dto(previous).model_dump_json()
            ),
        },
    )
    poisoned_key = (
        f"worker-plans:version-{current.plan_version.version_number}"
        if invalid_key == "worker"
        else f"flow-plans:{other_id.flow_id}"
    )
    await redis.set(poisoned_key, "wrong-type")

    with pytest.raises(ResponseError):
        await _publisher(redis).publish_plans(_worker_plans(flows), flows)

    assert await redis.hget(hash_key, active_field) == "3"
    # Redis may execute later staging commands despite an earlier command error.
    assert (
        await redis.hget(
            hash_key,
            f"version-{current.plan_version.version_number}",
        )
        is not None
    )


@pytest.mark.asyncio
async def test_worker_only_publication_is_not_ignored(redis: FakeRedis) -> None:
    worker = ServiceId()
    plan = _worker_plan(PlanVersion(7))

    await _publisher(redis).publish_plans({worker: plan}, {})

    payload = await redis.hget(
        f"worker-plans:version-{plan.plan_version.version_number}",
        str(worker.id),
    )
    assert WorkerPlanDto.model_validate_json(payload).plan_version == plan.plan_version
    assert await redis.keys("*") == [
        f"worker-plans:version-{plan.plan_version.version_number}"
    ]


@pytest.mark.asyncio
async def test_empty_publication_does_not_open_a_transaction() -> None:
    redis = AsyncMock(spec=asyncio_redis.Redis)

    await _publisher(redis).publish_plans({}, {})

    redis.pipeline.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["worker", "flow"])
async def test_prepares_every_payload_before_opening_a_transaction(kind: str) -> None:
    redis = AsyncMock(spec=asyncio_redis.Redis)
    flows = {FlowId(): _plan()}
    mapper, method = (
        (WorkerPlanMapper, "to_worker_plan_dto")
        if kind == "worker"
        else (FlowPlanMapper, "to_flow_plan_dto")
    )
    with (
        patch.object(mapper, method, side_effect=ValueError("serialization failed")),
        pytest.raises(ValueError, match="serialization failed"),
    ):
        await _publisher(redis).publish_plans(_worker_plans(flows), flows)

    redis.pipeline.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", [0, 1])
async def test_propagates_connection_errors_from_saving_or_activation(
    phase: int,
) -> None:
    redis = AsyncMock(spec=asyncio_redis.Redis)
    pipelines = [MagicMock(), MagicMock()]
    for pipeline in pipelines:
        pipeline.__aenter__.return_value = pipeline
        pipeline.__aexit__.return_value = False
        pipeline.execute = AsyncMock(return_value=[])
    pipelines[phase].execute.side_effect = RedisConnectionError("connection lost")
    redis.pipeline.side_effect = pipelines
    flows = {FlowId(): _plan()}

    with pytest.raises(RedisConnectionError):
        await _publisher(redis).publish_plans(_worker_plans(flows), flows)

    assert redis.pipeline.call_count == phase + 1
    pipelines[phase].execute.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_worker_reader_reads_published_address_and_reuses_cached_payload(
    redis: FakeRedis,
) -> None:
    from worker.adapters.outbound.plan_reader.redis_worker_plan_reader import (
        RedisWorkerPlanReader,
    )

    worker = ServiceId(id="worker")
    version = PlanVersion(8)
    await _publisher(redis).publish_plans({worker: _worker_plan(version)}, {})
    reader = RedisWorkerPlanReader(
        worker,
        redis,
        CachedRedisReader(HashMapRedisReader(), add_ttl_s=60.0),
        HashMapWorkerPlanAddressFactory(),
    )
    with patch.object(redis, "hget", wraps=redis.hget) as hget:
        for _ in range(2):
            assert (await reader.get_plan_by_version(version)).plan_version == version
        hget.assert_called_once_with("worker-plans:version-8", "worker")
