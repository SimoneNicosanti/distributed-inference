from abc import ABC, abstractmethod

from shared.plan.plan_version import PlanVersion


class PlanPreparationError(Exception):
    pass


class PlanApplicationError(Exception):
    pass


class PlanDeployer[PlanT](ABC):
    @abstractmethod
    async def prepare_plan(self, plan: PlanT) -> None: ...

    @abstractmethod
    async def apply_plan(self, plan_version: PlanVersion) -> None: ...
