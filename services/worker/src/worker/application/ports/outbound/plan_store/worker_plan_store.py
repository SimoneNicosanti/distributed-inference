from abc import ABC, abstractmethod

from shared.plan.plan_version import PlanVersion
from worker.domain.plan.worker.worker_plan import WorkerPlan


class WorkerPlanStore(ABC):
    @abstractmethod
    async def put_worker_plan(self, worker_plan: WorkerPlan) -> None: ...

    @abstractmethod
    async def get_worker_plan_by_version(
        self, version: PlanVersion
    ) -> WorkerPlan | None: ...

    @abstractmethod
    async def activate_worker_plan(self, version: PlanVersion) -> None: ...
