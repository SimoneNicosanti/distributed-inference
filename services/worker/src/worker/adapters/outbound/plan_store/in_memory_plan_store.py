from typing import override

from shared.plan.plan_version import PlanVersion
from worker.application.ports.outbound.plan_store.partition_execution_plan_reader import (
    PartitionExecutionPlanReader,
)
from worker.application.ports.outbound.plan_store.routing_plan_reader import (
    RoutingPlanReader,
)
from worker.application.ports.outbound.plan_store.scheduling_plan_reader import (
    SchedulingPlanReader,
)
from worker.application.ports.outbound.plan_store.worker_plan_store import (
    WorkerPlanStore,
)
from worker.domain.plan.partition_execution_plan import PartitionExecutionPlan
from worker.domain.plan.routing_plan import RoutingPlan
from worker.domain.plan.scheduling_plan import SchedulingPlan
from worker.domain.plan.worker_plan import WorkerPlan


class InMemoryPlanStore(
    WorkerPlanStore,
    SchedulingPlanReader,
    RoutingPlanReader,
    PartitionExecutionPlanReader,
):
    def __init__(self) -> None:
        self._latest_worker_plan: WorkerPlan | None = None
        self._active_worker_plan: WorkerPlan | None = None
        self._worker_plans_by_version: dict[PlanVersion, WorkerPlan] = {}

    @override
    async def put_worker_plan(self, worker_plan: WorkerPlan) -> None:
        existing_plan = self._worker_plans_by_version.get(worker_plan.plan_version)
        if existing_plan is not None:
            if existing_plan != worker_plan:
                raise ValueError(
                    f"Worker plan version {worker_plan.plan_version} already exists"
                )
            return

        if (
            self._latest_worker_plan is None
            or worker_plan.plan_version > self._latest_worker_plan.plan_version
        ):
            self._latest_worker_plan = worker_plan

        self._worker_plans_by_version[worker_plan.plan_version] = worker_plan

    @override
    async def get_worker_plan_by_version(
        self, version: PlanVersion
    ) -> WorkerPlan | None:
        return self._worker_plans_by_version.get(version)

    @override
    async def get_latest_worker_plan(self) -> WorkerPlan | None:
        return self._latest_worker_plan

    @override
    async def get_active_worker_plan(self) -> WorkerPlan | None:
        return self._active_worker_plan

    @override
    async def activate_worker_plan(self, version: PlanVersion) -> None:
        worker_plan = self._worker_plans_by_version.get(version)
        if worker_plan is None:
            raise ValueError(f"Worker plan version {version} does not exist")
        self._active_worker_plan = worker_plan

    @override
    async def get_scheduling_plan_by_version(
        self, version: PlanVersion
    ) -> SchedulingPlan | None:
        worker_plan = self._worker_plans_by_version.get(version)
        return worker_plan.scheduling_plan if worker_plan is not None else None

    @override
    async def get_routing_plan_by_version(
        self, version: PlanVersion
    ) -> RoutingPlan | None:
        worker_plan = self._worker_plans_by_version.get(version)
        return worker_plan.routing_plan if worker_plan is not None else None

    @override
    async def get_partition_execution_plan_by_version(
        self, version: PlanVersion
    ) -> PartitionExecutionPlan | None:
        worker_plan = self._worker_plans_by_version.get(version)
        return worker_plan.partition_execution_plan if worker_plan is not None else None
