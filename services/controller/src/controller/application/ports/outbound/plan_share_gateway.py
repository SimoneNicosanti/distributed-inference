from abc import ABC, abstractmethod

from controller.domain.plan.worker_plan import WorkerPlan
from shared.plan.plan_version import PlanVersion
from shared.service.service import WorkerId


class PlanShareGateway(ABC):
    @abstractmethod
    async def send_prepare_plan_message(
        self, worker_id: WorkerId, plan: WorkerPlan
    ) -> None: ...

    @abstractmethod
    async def send_apply_plan_message(
        self,
        worker_id: WorkerId,
        plan_version: PlanVersion,
    ) -> None: ...
