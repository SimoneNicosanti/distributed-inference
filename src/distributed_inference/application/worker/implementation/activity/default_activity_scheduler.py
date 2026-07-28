from asyncio import Future
from typing import Any, Tuple, override

from distributed_inference.application.scheduling.contracts.request_scheduler import (
    RequestScheduler,
)
from distributed_inference.application.worker.contracts.activity.activity_request import (
    ActivityRequest,
)


class DefaultActivityScheduler(RequestScheduler):
    def __init__(self) -> None:
        super().__init__()

    class _QueueRequest:
        request: ActivityRequest
        future: Future[Any]
        priority: int

    @override
    async def enqueue(self, request: ActivityRequest, future: Future[Any]) -> None:
        raise NotImplementedError

    @override
    async def dequeue(self) -> Tuple[ActivityRequest, Future[Any]]:
        raise NotImplementedError

    @override
    async def length(self) -> int:
        raise NotImplementedError

    pass
