from abc import ABC, abstractmethod

from shared.plan.plan_version import PlanVersion
from worker.domain.plan.ingress.ingress_plan import IngressPlan


class IngressPlanReader(ABC):
    @abstractmethod
    async def get_ingress_plan_by_version(
        self, version: PlanVersion
    ) -> IngressPlan | None: ...
