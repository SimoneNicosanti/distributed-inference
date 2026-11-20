from abc import ABC, abstractmethod

from shared.plan.hop import PlannedExecutableHopId
from shared.plan.plan_version import PlanVersion
from worker.application.ports.outbound.hop_execution.hop_executor import HopExecutor


class HopExecutorInitError(Exception): ...


class HopExecutorInitializer(ABC):
    @abstractmethod
    async def initialize(
        self, plan_version: PlanVersion, hop_id: PlannedExecutableHopId
    ) -> HopExecutor: ...
