from abc import ABC, abstractmethod
from contextlib import AbstractAsyncContextManager

from shared.plan.hop import PlannedExecutableHopId
from shared.plan.plan_version import PlanVersion
from worker.application.ports.outbound.hop_execution.hop_executor import (
    HopExecutor,
)


class HopExecutorRegistrationError(Exception):
    pass


class HopExecutorUnregistrationError(Exception):
    pass


class HopExecutionAcquisitionError(Exception):
    pass


class HopExecutorProvider(ABC):
    @abstractmethod
    def acquire_hop_executor(
        self, plan_version: PlanVersion, executable_hop_id: PlannedExecutableHopId
    ) -> AbstractAsyncContextManager[HopExecutor]: ...
