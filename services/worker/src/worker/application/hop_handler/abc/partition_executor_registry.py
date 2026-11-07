from abc import ABC, abstractmethod
from contextlib import AbstractAsyncContextManager

from shared.plan.hop import PlannedExecutableHopId
from shared.plan.plan_version import PlanVersion
from worker.application.ports.outbound.partition_execution.partition_executor import (
    PartitionExecutor,
)


class PartitionExecutorRegistrationError(Exception):
    pass


class PartitionExecutorUnregistrationError(Exception):
    pass


class PartitionExecutionAcquisitionError(Exception):
    pass


class PartitionExecutorRegistry(ABC):
    @abstractmethod
    async def register_partition_executor(
        self,
        plan_version: PlanVersion,
        executable_hop_id: PlannedExecutableHopId,
        partition_executor: PartitionExecutor,
    ) -> None: ...

    @abstractmethod
    async def unregister_partition_executor(
        self,
        plan_version: PlanVersion,
        executable_hop_id: PlannedExecutableHopId,
    ) -> PartitionExecutor: ...

    @abstractmethod
    def acquire_partition_executor(
        self, plan_version: PlanVersion, executable_hop_id: PlannedExecutableHopId
    ) -> AbstractAsyncContextManager[PartitionExecutor]: ...

    @abstractmethod
    async def check_partition_executor_exists(
        self, plan_version: PlanVersion, executable_hop_id: PlannedExecutableHopId
    ) -> bool: ...
