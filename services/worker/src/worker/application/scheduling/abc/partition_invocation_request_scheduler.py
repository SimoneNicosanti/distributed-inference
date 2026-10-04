from abc import ABC, abstractmethod
from asyncio import Future
from typing import Any, override

from scheduling.request_scheduler import RequestScheduler
from worker.domain.partition.partition_invocation import (
    PartitionInvocationRequest,
)


class PartitionInvocationRequestScheduler(
    RequestScheduler[PartitionInvocationRequest, Any], ABC
):
    @abstractmethod
    @override
    async def enqueue(
        self, request: PartitionInvocationRequest, future: Future[Any]
    ) -> None: ...

    @abstractmethod
    @override
    async def dequeue(self) -> tuple[PartitionInvocationRequest, Future[Any]]: ...

    @abstractmethod
    @override
    async def length(self) -> int: ...
