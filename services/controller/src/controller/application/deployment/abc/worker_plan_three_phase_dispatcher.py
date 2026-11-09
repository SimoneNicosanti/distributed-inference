from abc import ABC, abstractmethod

from controller.domain.plan.flow_plan import FlowPlan
from controller.domain.plan.worker_plan import WorkerPlan
from shared.flow.flow import FlowId
from shared.service.service import WorkerId


class PlanThreePhaseDispatcher(ABC):
    @abstractmethod
    async def dispatch_plans(
        self,
        flow_plans: dict[FlowId, FlowPlan],
        worker_plans: dict[WorkerId, WorkerPlan],
    ) -> None:
        pass
