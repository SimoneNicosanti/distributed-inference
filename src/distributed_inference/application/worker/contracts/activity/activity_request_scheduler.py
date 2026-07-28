from abc import ABC, abstractmethod
from asyncio import Future
from typing import Any, Tuple, override

from distributed_inference.application.scheduling.contracts.request_scheduler import (
    RequestScheduler,
)
from distributed_inference.application.worker.contracts.activity.activity_request import (
    ActivityRequest,
)


class ActivityRequestScheduler(RequestScheduler, ABC):
    @abstractmethod
    @override
    async def enqueue(self, request: ActivityRequest, future: Future[Any]) -> None: ...

    @abstractmethod
    @override
    async def dequeue(self) -> Tuple[ActivityRequest, Future[Any]]: ...

    @abstractmethod
    @override
    async def length(self) -> int: ...

    pass
