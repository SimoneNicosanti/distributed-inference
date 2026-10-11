from abc import ABC, abstractmethod

from shared.plan.plan_version import PlanVersion


class ActivePlanVersionReader(ABC):
    @abstractmethod
    async def get_active_plan_version(self) -> PlanVersion | None: ...
