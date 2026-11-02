from abc import ABC, abstractmethod

from shared.plan.hop import PlannedInvocableHopId
from shared.plan.plan_version import PlanVersion
from shared.service.service import WorkerId


class HopDeploymentDirectory(ABC):
    @abstractmethod
    async def get_worker_for_planned_invocable_hop(
        self,
        plan_version: PlanVersion,
        planned_hop: PlannedInvocableHopId,
    ) -> WorkerId: ...
