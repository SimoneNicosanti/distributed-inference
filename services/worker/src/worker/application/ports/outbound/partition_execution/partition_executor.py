from abc import ABC, abstractmethod

from worker.domain.partition.partition_execution import (
    PartitionExecutionInput,
    PartitionExecutionOutput,
)


## Executes a single model partition.
class PartitionExecutor(ABC):
    @abstractmethod
    async def execute(
        self, partition_execution_input: PartitionExecutionInput
    ) -> PartitionExecutionOutput: ...

    ## close method can be useful for cleaning up resources
    @abstractmethod
    async def close(self) -> None: ...
