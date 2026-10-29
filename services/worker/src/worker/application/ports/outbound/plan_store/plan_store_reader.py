from abc import ABC, abstractmethod

from shared.plan.plan_version import PlanVersion


class PlanStoreReader[T](ABC):
    @abstractmethod
    async def get_plan_by_version(self, version: PlanVersion) -> T: ...

    @abstractmethod
    async def get_active_plan_version(
        self,
    ) -> PlanVersion: ...
