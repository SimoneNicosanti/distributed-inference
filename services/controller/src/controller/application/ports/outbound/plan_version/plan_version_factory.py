from abc import ABC, abstractmethod

from shared.plan.plan_version import PlanVersion


class PlanVersionFactory(ABC):
    @abstractmethod
    async def build_next_plan_version(self) -> PlanVersion: ...
