from abc import ABC, abstractmethod
from asyncio import Future
from typing import Any, override

from scheduling.request_scheduler import RequestScheduler
from worker.domain.sub_model.invocation.sub_model_invocation_request_response import (
    SubModelInvocationRequest,
)


class SubModelInvocationRequestScheduler(
    RequestScheduler[SubModelInvocationRequest, Any], ABC
):
    @abstractmethod
    @override
    async def enqueue(
        self, request: SubModelInvocationRequest, future: Future[Any]
    ) -> None: ...

    @abstractmethod
    @override
    async def dequeue(self) -> tuple[SubModelInvocationRequest, Future[Any]]: ...

    @abstractmethod
    @override
    async def length(self) -> int: ...
