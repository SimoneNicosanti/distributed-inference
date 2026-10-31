from abc import ABC, abstractmethod

from front_end.domain.plan.flow_plan import FlowPlan
from front_end.domain.plan.front_end_plan import FrontEndPlan
from shared.flow.flow import FlowId


class FrontEndPlanReader(ABC):
    @abstractmethod
    async def get_current_front_end_plan(
        self,
    ) -> FrontEndPlan: ...

    @abstractmethod
    async def get_current_flow_plan(self, flow_id: FlowId) -> FlowPlan: ...
