from typing import override

from shared.plan.plan_version import PlanVersion
from worker.application.ports.outbound.plan_reader.worker_plan_reader import (
    WorkerPlanReader,
)
from worker.application.ports.outbound.plan_reader.worker_plan_view_readers import (
    DeploymentPlanReader,
    FailurePlanReader,
    HopPlacementPlanReader,
    HopRoutingPlanReader,
    PartitionExecutionPlanReader,
    SchedulingPlanReader,
)
from worker.domain.plan.deployment_plan import DeploymentPlan
from worker.domain.plan.failure_plan import FailurePlan
from worker.domain.plan.hop_placement_plan import HopPlacementPlan
from worker.domain.plan.hop_routing_plan import HopRoutingPlan
from worker.domain.plan.partition_execution_plan import PartitionExecutionPlan
from worker.domain.plan.scheduling_plan import SchedulingPlan


class AllWorkerPlanViewReader(
    SchedulingPlanReader,
    HopRoutingPlanReader,
    PartitionExecutionPlanReader,
    FailurePlanReader,
    DeploymentPlanReader,
    HopPlacementPlanReader,
):
    def __init__(self, worker_plan_reader: WorkerPlanReader) -> None:
        super().__init__()
        self._worker_plan_reader: WorkerPlanReader = worker_plan_reader

    @override
    async def get_scheduling_plan_by_version(
        self, version: PlanVersion
    ) -> SchedulingPlan:
        worker_plan = await self._worker_plan_reader.get_plan_by_version(version)
        return worker_plan.scheduling_plan

    @override
    async def get_hop_routing_plan_by_version(
        self, version: PlanVersion
    ) -> HopRoutingPlan:
        worker_plan = await self._worker_plan_reader.get_plan_by_version(version)
        return worker_plan.hop_routing_plan

    @override
    async def get_partition_execution_plan_by_version(
        self, version: PlanVersion
    ) -> PartitionExecutionPlan:
        worker_plan = await self._worker_plan_reader.get_plan_by_version(version)
        return worker_plan.partition_execution_plan

    @override
    async def get_failure_plan_by_version(self, version: PlanVersion) -> FailurePlan:
        worker_plan = await self._worker_plan_reader.get_plan_by_version(version)
        return worker_plan.failure_plan

    @override
    async def get_deployment_plan_by_version(
        self, version: PlanVersion
    ) -> DeploymentPlan:
        worker_plan = await self._worker_plan_reader.get_plan_by_version(version)
        return worker_plan.deployment_plan

    @override
    async def get_hop_placement_plan(self, version: PlanVersion) -> HopPlacementPlan:
        worker_plan = await self._worker_plan_reader.get_plan_by_version(version)
        return worker_plan.hop_placement_plan
