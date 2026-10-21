from abc import ABC, abstractmethod

from controller.domain.plan.service_plan import ServicePlan
from shared.plan.plan_version import PlanVersion
from shared.service.service import WorkerId


class PlanShareGateway(ABC):
    @abstractmethod
    async def send_prepare_plan_message(
        self, worker_id: WorkerId, plan: ServicePlan
    ) -> None: ...

    @abstractmethod
    async def send_apply_plan_message(
        self,
        worker_id: WorkerId,
        plan_version: PlanVersion,
    ) -> None: ...
