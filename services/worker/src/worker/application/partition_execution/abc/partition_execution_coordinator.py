from abc import ABC, abstractmethod

from worker.domain.partition.partition_invocation import (
    PartitionInvocationRequest,
    PartitionInvocationResult,
)


## This is the inference coordinator for a pool of inference workers
class PartitionExecutionCoordinator(ABC):
    @abstractmethod
    async def process_partition_invocation_request(
        self, partition_invocation_request: PartitionInvocationRequest
    ) -> PartitionInvocationResult: ...
