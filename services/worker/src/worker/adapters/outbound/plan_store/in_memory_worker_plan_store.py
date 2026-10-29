from typing import override

from shared.plan.plan_version import PlanVersion
from worker.adapters.outbound.plan_store.in_memory_plan_store import InMemoryPlanStore
from worker.application.ports.outbound.plan_store.worker_sub_plan_readers import (
    PartitionExecutionPlanReader,
    RoutingPlanReader,
    SchedulingPlanReader,
)
from worker.domain.plan.worker.partition_execution_plan import PartitionExecutionPlan
from worker.domain.plan.worker.routing_plan import RoutingPlan
from worker.domain.plan.worker.scheduling_plan import SchedulingPlan
from worker.domain.plan.worker.worker_plan import WorkerPlan


class InMemoryWorkerPlanStore(
    InMemoryPlanStore[WorkerPlan],
    SchedulingPlanReader,
    RoutingPlanReader,
    PartitionExecutionPlanReader,
):
    def __init__(self) -> None:
        super().__init__()

    @override
    async def get_scheduling_plan_by_version(
        self, version: PlanVersion
    ) -> SchedulingPlan:
        worker_plan = await self.get_plan_by_version(version)
        return worker_plan.scheduling_plan

    @override
    async def get_routing_plan_by_version(self, version: PlanVersion) -> RoutingPlan:
        worker_plan = await self.get_plan_by_version(version)
        return worker_plan.routing_plan

    @override
    async def get_partition_execution_plan_by_version(
        self, version: PlanVersion
    ) -> PartitionExecutionPlan:
        worker_plan = await self.get_plan_by_version(version)
        return worker_plan.partition_execution_plan
