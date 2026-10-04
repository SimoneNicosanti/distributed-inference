from abc import ABC, abstractmethod

from worker.domain.context.partition_invocation_context import (
    PartitionInvocationId,
)


class PartitionInvocationCompletionRegistry(ABC):
    @abstractmethod
    async def wait_for_partition_invocation_completion(
        self, partition_invocation_id: PartitionInvocationId
    ) -> None: ...

    @abstractmethod
    async def register_partition_invocation_success(
        self, partition_invocation_id: PartitionInvocationId
    ) -> None: ...

    @abstractmethod
    async def register_partition_invocation_failure(
        self, partition_invocation_id: PartitionInvocationId, exception: BaseException
    ) -> None: ...
