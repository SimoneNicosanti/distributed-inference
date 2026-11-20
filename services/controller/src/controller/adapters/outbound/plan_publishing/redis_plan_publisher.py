from typing import override

from redis import asyncio as asyncio_redis

from controller.adapters.outbound.plan_publishing.mappers.flow_plan_mapper import (
    FlowPlanMapper,
)
from controller.adapters.outbound.plan_publishing.mappers.worker_plan_mapper import (
    WorkerPlanMapper,
)
from controller.application.ports.outbound.plan_publishing.plan_publisher import (
    PlanPublisher,
)
from controller.domain.plan.flow_plan import FlowPlan
from controller.domain.plan.worker_plan import WorkerPlan
from integration.redis.front_end_plan.keys_factory import FlowPlanKeyFactory
from integration.redis.worker_plan.keys_factory import WorkerPlanKeyFactory
from shared.flow.flow import FlowId
from shared.service.service import WorkerId


class RedisPlanPublisher(PlanPublisher):
    def __init__(
        self,
        redis: asyncio_redis.Redis,
        worker_plan_key_factory: WorkerPlanKeyFactory,
        flow_plan_key_factory: FlowPlanKeyFactory,
    ) -> None:
        self._redis = redis
        self._flow_plan_key_factory = flow_plan_key_factory
        self._worker_plan_key_factory = worker_plan_key_factory

    @override
    async def publish_plans(
        self,
        worker_plans: dict[WorkerId, WorkerPlan],
        flow_plans: dict[FlowId, FlowPlan],
    ) -> None:
        entries: list[tuple[str, str, str]] = []
        activations: list[tuple[str, str, str]] = []
        for worker_id, worker_plan in worker_plans.items():
            payload = WorkerPlanMapper.to_worker_plan_dto(worker_plan).model_dump_json()
            hash_key = self._worker_plan_key_factory.build_hash_key(
                worker_plan.plan_version
            )
            field_key = self._worker_plan_key_factory.build_field_key(worker_id)
            entries.append((hash_key, field_key, payload))

        for flow_id, flow_plan in flow_plans.items():
            payload = FlowPlanMapper.to_flow_plan_dto(flow_plan).model_dump_json()
            hash_key = await self._flow_plan_key_factory.build_hash_key(flow_id)
            version_key = await self._flow_plan_key_factory.build_plan_field_key(
                flow_plan.plan_version
            )
            active_field = (
                await self._flow_plan_key_factory.build_current_plan_field_key()
            )
            entries.append((hash_key, version_key, payload))
            activations.append(
                (hash_key, active_field, str(flow_plan.plan_version.version_number))
            )

        if not entries:
            return

        async with self._redis.pipeline(transaction=True) as pipeline:
            for hash_key, field_key, payload in entries:
                pipeline.hset(hash_key, field_key, payload)
            await pipeline.execute()

        if activations:
            async with self._redis.pipeline(transaction=True) as pipeline:
                for hash_key, active_field, version in activations:
                    pipeline.hset(hash_key, active_field, version)
                await pipeline.execute()
