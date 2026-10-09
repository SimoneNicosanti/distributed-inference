from abc import ABC, abstractmethod

from shared.plan.plan_version import PlanVersion
from worker.domain.plan.scheduling_plan import SchedulingPlan


class SchedulingPlanReader(ABC):
    @abstractmethod
    async def get_scheduling_plan_by_version(
        self, version: PlanVersion
    ) -> SchedulingPlan | None: ...
