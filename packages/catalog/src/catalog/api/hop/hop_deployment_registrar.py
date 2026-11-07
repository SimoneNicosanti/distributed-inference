from abc import ABC, abstractmethod

from shared.plan.hop import PlannedInvocableHopId
from shared.plan.plan_version import PlanVersion
from shared.service.service import WorkerId


class HopDeploymentRegistrationError(Exception):
    pass


class HopDeploymentRegistrar(ABC):
    @abstractmethod
    async def register_worker_for_planned_invocable_hops[
        PlannedHopIdT: PlannedInvocableHopId
    ](
        self,
        plan_version: PlanVersion,
        planned_hops: tuple[PlannedHopIdT, ...],
        worker_id: WorkerId,
    ) -> None: ...
