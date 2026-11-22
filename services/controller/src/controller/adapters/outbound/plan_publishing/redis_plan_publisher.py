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
from redis_access.address_factory.front_end_plan.address_factory import (
    FlowPlanAddressFactory,
)
from redis_access.address_factory.worker_plan.address_factory import (
    WorkerPlanAddressFactory,
)
from redis_access.contract.redis_address import HashMapRedisAddress
from shared.flow.flow import FlowId
from shared.service.service import WorkerId


class RedisPlanPublisher(PlanPublisher):
    def __init__(
        self,
        redis: asyncio_redis.Redis,
        worker_plan_address_factory: WorkerPlanAddressFactory[HashMapRedisAddress],
        flow_plan_address_factory: FlowPlanAddressFactory[HashMapRedisAddress],
    ) -> None:
        self._redis = redis
        self._flow_plan_address_factory = flow_plan_address_factory
        self._worker_plan_address_factory = worker_plan_address_factory

    @override
    async def publish_plans(
        self,
        worker_plans: dict[WorkerId, WorkerPlan],
        flow_plans: dict[FlowId, FlowPlan],
    ) -> None:
        entries: list[tuple[HashMapRedisAddress, str]] = []
        activations: list[tuple[HashMapRedisAddress, str]] = []
        for worker_id, worker_plan in worker_plans.items():
            payload = WorkerPlanMapper.to_worker_plan_dto(worker_plan).model_dump_json()
            addr = (
                self._worker_plan_address_factory.build_versioned_worker_plan_address(
                    worker_plan.plan_version, worker_id
                )
            )
            entries.append((addr, payload))

        for flow_id, flow_plan in flow_plans.items():
            payload = FlowPlanMapper.to_flow_plan_dto(flow_plan).model_dump_json()
            addr = (
                await self._flow_plan_address_factory.build_versioned_flow_plan_address(
                    flow_plan.plan_version, flow_id
                )
            )
            active_addr = (
                await self._flow_plan_address_factory.build_flow_active_plan_version(
                    flow_id
                )
            )
            entries.append((addr, payload))
            activations.append(
                (active_addr, str(flow_plan.plan_version.version_number))
            )

        if not entries:
            return

        async with self._redis.pipeline(transaction=True) as pipeline:
            for addr, payload in entries:
                pipeline.hset(addr.hash_map_key, addr.field_key, payload)
            await pipeline.execute()

        if activations:
            async with self._redis.pipeline(transaction=True) as pipeline:
                for addr, version in activations:
                    pipeline.hset(addr.hash_map_key, addr.field_key, version)
                await pipeline.execute()
