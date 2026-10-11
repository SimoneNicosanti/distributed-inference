from abc import abstractmethod

from shared.plan.plan_version import PlanVersion
from worker.application.ports.outbound.plan_store.ingress_plan_reader import (
    IngressPlanReader,
)
from worker.domain.plan.ingress.ingress_plan import IngressPlan


class IngressPlanStore(IngressPlanReader):
    @abstractmethod
    async def put_ingress_plan(self, ingress_plan: IngressPlan) -> None: ...

    @abstractmethod
    async def activate_ingress_plan(self, version: PlanVersion) -> None: ...
