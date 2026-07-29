from abc import ABC, abstractmethod
from asyncio import Future
from typing import Any, Tuple, override

from distributed_inference.activity_manager.domain.activity_request import (
    ActivityRequest,
)
from distributed_inference.building_blocks.scheduling.request_scheduler import (
    RequestScheduler,
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
