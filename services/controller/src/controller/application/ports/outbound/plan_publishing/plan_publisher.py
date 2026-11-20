from abc import ABC, abstractmethod

from controller.domain.plan.flow_plan import FlowPlan
from controller.domain.plan.worker_plan import WorkerPlan
from shared.flow.flow import FlowId
from shared.service.service import WorkerId


class PlanPublisher(ABC):
    @abstractmethod
    async def publish_plans(
        self,
        worker_plans: dict[WorkerId, WorkerPlan],
        flow_plans: dict[FlowId, FlowPlan],
    ) -> None: ...
