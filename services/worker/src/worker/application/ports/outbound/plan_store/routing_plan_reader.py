from abc import ABC, abstractmethod

from shared.plan.plan_version import PlanVersion
from worker.domain.plan.routing_plan import RoutingPlan


class RoutingPlanReader(ABC):
    @abstractmethod
    async def get_routing_plan_by_version(
        self, version: PlanVersion
    ) -> RoutingPlan | None: ...
