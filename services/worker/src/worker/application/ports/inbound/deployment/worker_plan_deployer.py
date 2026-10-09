from abc import ABC, abstractmethod

from worker.domain.plan.worker_plan import WorkerPlan


class WorkerPlanDeployer(ABC):
    @abstractmethod
    async def prepare_worker_plan(self, worker_plan: WorkerPlan) -> None: ...

    @abstractmethod
    async def apply_worker_plan(self, worker_plan: WorkerPlan) -> None: ...
