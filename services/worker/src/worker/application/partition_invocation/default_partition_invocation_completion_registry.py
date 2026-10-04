import asyncio
from typing import override

from worker.application.partition_invocation.abc.partition_invocation_completion_registry import (
    PartitionInvocationCompletionRegistry,
)
from worker.domain.context.partition_invocation_context import (
    PartitionInvocationId,
)

## TODO asyncio.shield might be useful also for other things!


class DefaultPartitionInvocationCompletionRegistry(
    PartitionInvocationCompletionRegistry
):
    def __init__(self) -> None:
        self._completion_events: dict[PartitionInvocationId, asyncio.Future[None]] = {}

    @override
    async def wait_for_partition_invocation_completion(
        self, partition_invocation_id: PartitionInvocationId
    ) -> None:
        if partition_invocation_id not in self._completion_events:
            future = asyncio.get_running_loop().create_future()
            self._completion_events[partition_invocation_id] = future
        await asyncio.shield(self._completion_events[partition_invocation_id])

    @override
    async def register_partition_invocation_success(
        self, partition_invocation_id: PartitionInvocationId
    ) -> None:
        if partition_invocation_id in self._completion_events:
            self._completion_events[partition_invocation_id].set_result(None)
            self._completion_events.pop(partition_invocation_id)

    @override
    async def register_partition_invocation_failure(
        self, partition_invocation_id: PartitionInvocationId, exception: BaseException
    ) -> None:
        if partition_invocation_id in self._completion_events:
            self._completion_events[partition_invocation_id].set_exception(exception)
        pass
