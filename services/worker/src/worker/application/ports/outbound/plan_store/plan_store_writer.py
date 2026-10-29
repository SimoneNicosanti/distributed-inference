from abc import ABC, abstractmethod

from shared.plan.plan_version import PlanVersion


class PlanStoreWriter[PlanT](ABC):
    @abstractmethod
    async def put_plan(self, plan: PlanT) -> None: ...

    @abstractmethod
    async def activate_plan(self, version: PlanVersion) -> None: ...
