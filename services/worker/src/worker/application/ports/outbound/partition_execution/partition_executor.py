from abc import ABC, abstractmethod

from shared.plan.hop_logical_id import PlannedPartitionReplicaId
from worker.domain.hop.hop_execution import (
    HopExecutionInput,
    HopExecutionOutput,
)


## Executes a single model partition.
class PartitionExecutor(ABC):
    @abstractmethod
    async def execute(
        self, partition_execution_input: HopExecutionInput[PlannedPartitionReplicaId]
    ) -> HopExecutionOutput[PlannedPartitionReplicaId]: ...

    ## close method can be useful for cleaning up resources
    @abstractmethod
    async def close(self) -> None: ...
