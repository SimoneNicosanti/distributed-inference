from abc import ABC, abstractmethod
from collections.abc import Mapping

from controller.domain.plan.front_end_plan import FlowPlan
from shared.flow.flow import FlowId


class FlowPlanPublisher(ABC):
    @abstractmethod
    async def publish_flow_plans(
        self, flow_plans: Mapping[FlowId, FlowPlan]
    ) -> None: ...
