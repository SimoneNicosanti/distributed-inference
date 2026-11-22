from typing import override

from redis import asyncio as asyncio_redis

from front_end.adapters.outbound.mappers.flow_plan_dto_mapper import FlowPlanDtoMapper
from front_end.application.ports.outbound.inference.flow_plan_reader import (
    FlowPlanReader,
    FlowPlanReadError,
)
from front_end.domain.flow_plan import FlowPlan
from integration.dto.flow_plan.flow_plan_dto import FlowPlanDto
from redis_access.address_factory.front_end_plan.address_factory import (
    FlowPlanAddressFactory,
)
from redis_access.contract.redis_address import HashMapRedisAddress
from redis_access.redis_reader.abc.redis_reader import RedisReader, RedisReadError
from shared.flow.flow import FlowId
from shared.plan.plan_version import PlanVersion


class RedisFlowPlanReader(FlowPlanReader):
    def __init__(
        self,
        redis: asyncio_redis.Redis,
        current_version_reader: RedisReader[HashMapRedisAddress],
        versioned_plan_reader: RedisReader[HashMapRedisAddress],
        address_factory: FlowPlanAddressFactory[HashMapRedisAddress],
    ) -> None:
        super().__init__()
        self._redis = redis
        self._current_version_reader = current_version_reader
        self._versioned_plan_reader = versioned_plan_reader
        self._address_factory = address_factory

    @override
    async def get_current_plan_version_for_flow(self, flow_id: FlowId) -> PlanVersion:
        try:
            addr = await self._address_factory.build_flow_active_plan_version(flow_id)
            value = await self._current_version_reader.read(self._redis, addr)
            return PlanVersion(version_number=int(value))
        except (RedisReadError, ValueError) as e:
            raise FlowPlanReadError(
                f"Could not read current flow plan for flow {flow_id}"
            ) from e

    @override
    async def get_flow_plan_by_plan_version(
        self, flow_id: FlowId, plan_version: PlanVersion
    ) -> FlowPlan:
        try:
            addr = await self._address_factory.build_versioned_flow_plan_address(
                plan_version, flow_id
            )
            value = await self._versioned_plan_reader.read(self._redis, addr)
            dto = FlowPlanDto.model_validate_json(value)
            if dto.plan_version != plan_version:
                raise ValueError(
                    "Stored flow plan version does not match requested version"
                )
            return FlowPlanDtoMapper.to_flow_plan(dto)
        except (RedisReadError, ValueError) as e:
            raise FlowPlanReadError(
                f"Could not read plan version {plan_version} for flow {flow_id}"
            ) from e
