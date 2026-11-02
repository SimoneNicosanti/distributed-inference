from typing import override

from redis import asyncio as asyncio_redis

from front_end.adapters.outbound.mappers.flow_plan_dto_mapper import FlowPlanDtoMapper
from front_end.application.ports.outbound.inference.flow_plan_reader import (
    FlowPlanReader,
)
from front_end.domain.flow_plan import FlowPlan
from integration.dto.flow_plan.flow_plan_dto import FlowPlanDto
from integration.redis.front_end_plan.keys_factory import FlowPlanKeyFactory
from shared.flow.flow import FlowId
from shared.plan.plan_version import PlanVersion


class RedisFlowPlanReader(FlowPlanReader):
    def __init__(
        self, redis: asyncio_redis.Redis, key_factory: FlowPlanKeyFactory
    ) -> None:
        super().__init__()
        self._redis = redis
        self._key_factory = key_factory

    @override
    async def get_current_plan_version_for_flow(self, flow_id: FlowId) -> PlanVersion:
        key = await self._key_factory.build_current_plan_key(flow_id)

        value = await self._redis.get(key)
        if value is None:
            raise KeyError(f"No plan for flow id {flow_id.flow_id}")

        return PlanVersion(version_number=int(value))

    @override
    async def get_flow_plan_by_plan_version(
        self, flow_id: FlowId, plan_version: PlanVersion
    ) -> FlowPlan:
        dict_key = await self._key_factory.build_flow_plans_dict_key(flow_id)
        dict_elem_key = await self._key_factory.build_flow_plan_elem_key(plan_version)

        value = await self._redis.hget(dict_key, dict_elem_key)
        if value is None:
            raise KeyError(
                f"No plan with version {plan_version.version_number} for flow id {flow_id}"
            )

        dto = FlowPlanDto.model_validate_json(value)
        if dto.plan_version != plan_version:
            raise ValueError(
                "Stored flow plan version does not match requested version"
            )
        flow_plan = FlowPlanDtoMapper.to_flow_plan(dto)

        return flow_plan
