from abc import ABC, abstractmethod

from shared.plan.plan_version import PlanVersion
from worker.domain.plan.worker_plan import WorkerPlan


class WorkerPlanReader(ABC):
    @abstractmethod
    async def get_plan_by_version(self, version: PlanVersion) -> WorkerPlan: ...


class WorkerPlanReadError(Exception):
    pass
