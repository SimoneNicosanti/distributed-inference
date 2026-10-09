from abc import ABC, abstractmethod
from contextlib import AbstractAsyncContextManager

from shared.plan.partition_replica_id import PartitionReplicaId
from worker.application.ports.outbound.partition_execution.partition_executor import (
    PartitionExecutor,
)


class PartitionExecutorRegistry(ABC):
    @abstractmethod
    async def register_partition_executor(
        self,
        replica_id: PartitionReplicaId,
        partition_executor: PartitionExecutor,
    ) -> None: ...

    @abstractmethod
    async def unregister_partition_executor(
        self,
        replica_id: PartitionReplicaId,
    ) -> PartitionExecutor: ...

    @abstractmethod
    def acquire_partition_executor(
        self, replica_id: PartitionReplicaId
    ) -> AbstractAsyncContextManager[PartitionExecutor]: ...

    @abstractmethod
    async def check_partition_executor_exists(
        self, replica_id: PartitionReplicaId
    ) -> bool: ...
