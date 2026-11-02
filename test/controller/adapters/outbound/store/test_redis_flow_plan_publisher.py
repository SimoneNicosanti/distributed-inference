from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from redis import asyncio as asyncio_redis
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import ResponseError

from controller.adapters.outbound.store.mappers.flow_plan_mapper import FlowPlanMapper
from controller.adapters.outbound.store.redis_flow_plan_publisher import (
    RedisFlowPlanPublisher,
)
from controller.application.ports.outbound.store.flow_plan_publisher import (
    FlowPlanPublisher,
)
from controller.domain.plan.front_end_plan import FlowPlan
from controller.domain.profile.model_variant_static_profile import TensorProperty
from front_end.adapters.outbound.inference.redis_flow_plan_reader import (
    RedisFlowPlanReader,
)
from integration.dto.flow_plan.flow_plan_dto import FlowPlanDto
from integration.redis.front_end_plan.keys_factory import StringFlowPlanKeyFactory
from shared.flow.flow import FlowId
from shared.model.model import ModelId
from shared.model.model_variant import ModelVariantId
from shared.plan.hop_logical_id import PlannedIngressReplicaId, PlannedSinkReplicaId
from shared.plan.plan_version import PlanVersion
from shared.user.user import UserId


def _plan(version: int = 3, variant_tag: str = "variant") -> FlowPlan:
    variant = ModelVariantId(ModelId("model"), variant_tag)
    return FlowPlan(
        plan_version=PlanVersion(version_number=version),
        model_variant_id=variant,
        input_tensor_properties={
            "scale": TensorProperty("float32", 4, ()),
            "features": TensorProperty("float32", 4, ("batch", 3)),
            "mask": TensorProperty("bool", 1, ("batch", 0)),
        },
        ingress_replica_id=PlannedIngressReplicaId(variant, 2),
        sink_replica_id=PlannedSinkReplicaId(variant, 5),
    )


@pytest.fixture
def redis_pipeline() -> tuple[AsyncMock, MagicMock]:
    redis = AsyncMock(spec=asyncio_redis.Redis)
    pipeline = MagicMock()
    pipeline.__aenter__.return_value = pipeline
    pipeline.__aexit__.return_value = False
    pipeline.execute = AsyncMock(return_value=[])
    redis.pipeline.return_value = pipeline
    return redis, pipeline


def _reader_from_writes(
    redis: AsyncMock, pipeline: MagicMock, factory: StringFlowPlanKeyFactory
) -> RedisFlowPlanReader:
    versions: dict[str, str] = {
        command.args[0]: command.args[1] for command in pipeline.set.call_args_list
    }
    payloads: dict[tuple[str, str], str] = {
        (command.args[0], command.args[1]): command.args[2]
        for command in pipeline.hset.call_args_list
    }
    redis.get = AsyncMock(side_effect=versions.get)
    redis.hget = AsyncMock(side_effect=lambda key, field: payloads.get((key, field)))
    return RedisFlowPlanReader(redis, factory)


def test_maps_model_contract_version_and_endpoint_identity() -> None:
    plan = _plan()

    dto = FlowPlanMapper.to_flow_plan_dto(plan)
    received = FlowPlanDto.model_validate_json(dto.model_dump_json())

    assert received.plan_version == plan.plan_version
    assert received.model_specs.model_variant_id == plan.model_variant_id
    assert received.ingress_replica_id == plan.ingress_replica_id
    assert received.sink_replica_id == plan.sink_replica_id
    assert {
        spec.name: (spec.shape, spec.dtype) for spec in received.model_specs.inputs
    } == {
        "features": (("batch", 3), "float32"),
        "mask": (("batch", 0), "bool"),
        "scale": ((), "float32"),
    }


@pytest.mark.asyncio
async def test_publishes_all_flows_in_one_transaction_using_integration_keys(
    redis_pipeline: tuple[AsyncMock, MagicMock],
) -> None:
    redis, pipeline = redis_pipeline
    factory = StringFlowPlanKeyFactory()
    plans = {
        FlowId(user_id=UserId()): _plan(),
        FlowId(user_id=UserId()): _plan(variant_tag="other"),
    }
    publisher: FlowPlanPublisher = RedisFlowPlanPublisher(redis, factory)

    await publisher.publish_flow_plans(plans)

    redis.pipeline.assert_called_once_with(transaction=True)
    assert pipeline.hset.call_count == len(plans)
    assert pipeline.set.call_count == len(plans)
    for flow_id, plan in plans.items():
        pipeline.hset.assert_any_call(
            await factory.build_flow_plans_dict_key(flow_id),
            await factory.build_flow_plan_elem_key(plan.plan_version),
            FlowPlanMapper.to_flow_plan_dto(plan).model_dump_json(),
        )
        pipeline.set.assert_any_call(
            await factory.build_current_plan_key(flow_id),
            str(plan.plan_version.version_number),
        )
    pipeline.execute.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_published_payloads_can_be_read_by_the_existing_front_end_reader(
    redis_pipeline: tuple[AsyncMock, MagicMock],
) -> None:
    redis, pipeline = redis_pipeline
    factory = StringFlowPlanKeyFactory()
    plans = {
        FlowId(user_id=UserId()): _plan(),
        FlowId(user_id=UserId()): _plan(4, "other"),
    }
    await RedisFlowPlanPublisher(redis, factory).publish_flow_plans(plans)
    reader = _reader_from_writes(redis, pipeline, factory)

    for flow_id, expected in plans.items():
        version = await reader.get_current_plan_version_for_flow(flow_id)
        received = await reader.get_flow_plan_by_plan_version(flow_id, version)

        assert version == expected.plan_version
        assert received.model_specs.model_variant_id == expected.model_variant_id
        assert received.ingress_replica_id == expected.ingress_replica_id
        assert received.sink_replica_id == expected.sink_replica_id
        assert {
            spec.name: (spec.shape, spec.dtype) for spec in received.model_specs.inputs
        } == {
            name: (property_.shape_expression, property_.data_type)
            for name, property_ in expected.input_tensor_properties.items()
        }


@pytest.mark.asyncio
async def test_keeps_previous_versions_when_publishing_a_new_current_version(
    redis_pipeline: tuple[AsyncMock, MagicMock],
) -> None:
    redis, pipeline = redis_pipeline
    factory = StringFlowPlanKeyFactory()
    publisher = RedisFlowPlanPublisher(redis, factory)
    flow_id = FlowId(user_id=UserId())
    previous, current = _plan(3), _plan(4)

    await publisher.publish_flow_plans({flow_id: previous})
    await publisher.publish_flow_plans({flow_id: current})
    reader = _reader_from_writes(redis, pipeline, factory)

    assert (
        await reader.get_current_plan_version_for_flow(flow_id) == current.plan_version
    )
    received = await reader.get_flow_plan_by_plan_version(
        flow_id, previous.plan_version
    )
    assert received.plan_version == previous.plan_version
    assert received.ingress_replica_id == previous.ingress_replica_id


@pytest.mark.asyncio
async def test_empty_publication_does_not_open_a_transaction(
    redis_pipeline: tuple[AsyncMock, MagicMock],
) -> None:
    redis, pipeline = redis_pipeline

    await RedisFlowPlanPublisher(redis, StringFlowPlanKeyFactory()).publish_flow_plans(
        {}
    )

    redis.pipeline.assert_not_called()
    pipeline.execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_prepares_every_payload_before_opening_a_transaction(
    redis_pipeline: tuple[AsyncMock, MagicMock],
) -> None:
    redis, pipeline = redis_pipeline
    first = _plan()
    plans = {
        FlowId(user_id=UserId()): first,
        FlowId(user_id=UserId()): _plan(variant_tag="other"),
    }
    with (
        patch.object(
            FlowPlanMapper,
            "to_flow_plan_dto",
            side_effect=[
                FlowPlanMapper.to_flow_plan_dto(first),
                ValueError("serialization failed"),
            ],
        ),
        pytest.raises(ValueError, match="serialization failed"),
    ):
        await RedisFlowPlanPublisher(
            redis, StringFlowPlanKeyFactory()
        ).publish_flow_plans(plans)

    redis.pipeline.assert_not_called()
    pipeline.execute.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error", [RedisConnectionError("connection lost"), ResponseError("write failed")]
)
async def test_propagates_transaction_errors(
    redis_pipeline: tuple[AsyncMock, MagicMock],
    error: Exception,
) -> None:
    redis, pipeline = redis_pipeline
    pipeline.execute.side_effect = error

    with pytest.raises(type(error)):
        await RedisFlowPlanPublisher(
            redis, StringFlowPlanKeyFactory()
        ).publish_flow_plans({FlowId(user_id=UserId()): _plan()})

    pipeline.execute.assert_awaited_once_with()
