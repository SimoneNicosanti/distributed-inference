from abc import ABC, abstractmethod
from asyncio import Future
from typing import Any, Tuple, override

from distributed_inference.building_blocks.scheduling.request_scheduler import (
    RequestScheduler,
)
from distributed_inference.worker.domain.sub_model.invocation.sub_model_invocation_request_response import (
    SubModelInvocationRequest,
)


class SubModelInvocationRequestScheduler(RequestScheduler, ABC):
    class QueueSubModelInvocationRequest(RequestScheduler.QueueRequest):
        pass

    @abstractmethod
    @override
    async def enqueue(
        self, request: SubModelInvocationRequest, future: Future[Any]
    ) -> None: ...

    @abstractmethod
    @override
    async def dequeue(self) -> Tuple[SubModelInvocationRequest, Future[Any]]: ...

    @abstractmethod
    @override
    async def length(self) -> int: ...
