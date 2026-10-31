from abc import ABC, abstractmethod
from contextlib import AbstractAsyncContextManager

from worker.application.ports.outbound.partition_execution.partition_executor import (
    PartitionExecutor,
)
from worker.domain.hop.hop import ExecutableHop


class PartitionExecutorRegistry(ABC):
    @abstractmethod
    async def register_partition_executor(
        self,
        executable_hop: ExecutableHop,
        partition_executor: PartitionExecutor,
    ) -> None: ...

    @abstractmethod
    async def unregister_partition_executor(
        self,
        executable_hop: ExecutableHop,
    ) -> PartitionExecutor: ...

    @abstractmethod
    def acquire_partition_executor(
        self, executable_hop: ExecutableHop
    ) -> AbstractAsyncContextManager[PartitionExecutor]: ...

    @abstractmethod
    async def check_partition_executor_exists(
        self, executable_hop: ExecutableHop
    ) -> bool: ...
