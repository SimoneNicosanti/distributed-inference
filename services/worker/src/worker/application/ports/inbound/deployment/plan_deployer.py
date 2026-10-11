from abc import ABC, abstractmethod

from shared.plan.plan_version import PlanVersion


class PlanDeployer[PlanT](ABC):
    @abstractmethod
    async def prepare_plan(self, plan: PlanT) -> None: ...

    @abstractmethod
    async def apply_plan(self, plan_version: PlanVersion) -> None: ...
