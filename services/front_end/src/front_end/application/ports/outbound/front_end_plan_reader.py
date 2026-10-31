from abc import ABC, abstractmethod

from front_end.domain.plan.flow_plan import FlowPlan
from shared.flow.flow import FlowId
from shared.plan.plan_version import PlanVersion


class FrontEndPlanReader(ABC):
    @abstractmethod
    async def get_current_plan_version(
        self,
    ) -> PlanVersion: ...

    @abstractmethod
    async def get_flow_plan_by_version(
        self, flow_id: FlowId, plan_version: PlanVersion
    ) -> FlowPlan: ...
