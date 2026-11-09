from collections.abc import Mapping
from typing import override

from redis import asyncio as asyncio_redis

from controller.adapters.outbound.store.mappers.flow_plan_mapper import FlowPlanMapper
from controller.application.ports.outbound.store.flow_plan_publisher import (
    FlowPlanPublisher,
)
from controller.domain.plan.flow_plan import FlowPlan
from integration.redis.front_end_plan.keys_factory import FlowPlanKeyFactory
from shared.flow.flow import FlowId


class RedisFlowPlanPublisher(FlowPlanPublisher):
    def __init__(
        self, redis: asyncio_redis.Redis, key_factory: FlowPlanKeyFactory
    ) -> None:
        self._redis = redis
        self._key_factory = key_factory

    @override
    async def publish_flow_plans(self, flow_plans: Mapping[FlowId, FlowPlan]) -> None:
        entries: list[tuple[str, str, str, str, str]] = []
        for flow_id, plan in flow_plans.items():
            payload = FlowPlanMapper.to_flow_plan_dto(plan).model_dump_json()
            versions_key = await self._key_factory.build_flow_plans_dict_key(flow_id)
            version_key = await self._key_factory.build_flow_plan_elem_key(
                plan.plan_version
            )
            active_key = await self._key_factory.build_current_plan_key(flow_id)
            entries.append(
                (
                    versions_key,
                    version_key,
                    active_key,
                    payload,
                    str(plan.plan_version.version_number),
                )
            )

        if not entries:
            return

        async with self._redis.pipeline(transaction=True) as pipeline:
            for versions_key, version_key, active_key, payload, version in entries:
                pipeline.hset(versions_key, version_key, payload)
                pipeline.set(active_key, version)
            await pipeline.execute()
