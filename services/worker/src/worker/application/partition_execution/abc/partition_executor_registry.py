from abc import ABC, abstractmethod
from contextlib import AbstractAsyncContextManager

from shared.plan.plan import PartitionDeployment
from worker.application.ports.outbound.partition_executor import (
    PartitionExecutor,
)


class PartitionExecutorRegistry(ABC):
    @abstractmethod
    async def register_partition_executor(
        self,
        partition_deployment: PartitionDeployment,
        partition_executor: PartitionExecutor,
    ) -> None: ...

    @abstractmethod
    async def unregister_partition_executor(
        self,
        partition_deployment: PartitionDeployment,
    ) -> PartitionExecutor: ...

    @abstractmethod
    def acquire_partition_executor(
        self, partition_deployment: PartitionDeployment
    ) -> AbstractAsyncContextManager[PartitionExecutor]: ...

    @abstractmethod
    async def check_partition_executor_exists(
        self, partition_deployment: PartitionDeployment
    ) -> bool: ...
