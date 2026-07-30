import asyncio
import itertools
import time
from asyncio import Future
from dataclasses import dataclass
from typing import Any, Tuple, override

from distributed_inference.activity_manager.application.scheduling.contracts.activity_request_scheduler import (
    ActivityRequestScheduler,
)
from distributed_inference.activity_manager.application.scheduling.contracts.activity_request_static_priority_assigner import (
    ActivityRequestStaticPriorityAssigner,
)
from distributed_inference.activity_manager.domain.activity_request import (
    ActivityRequest,
)
from distributed_inference.building_blocks.scheduling.request_scheduler import (
    RequestScheduler,
)


class DefaultActivityRequestScheduler(ActivityRequestScheduler):
    @dataclass
    class QueueActivityRequest(RequestScheduler.QueueRequest):
        priority: int
        sequence: int

        def __lt__(self, other: Any) -> bool:
            if not isinstance(
                other, DefaultActivityRequestScheduler.QueueActivityRequest
            ):
                return NotImplemented

            return (self.priority, self.timestamp, self.sequence) < (
                other.priority,
                other.timestamp,
                other.sequence,
            )

    def __init__(
        self, priority_assigner: ActivityRequestStaticPriorityAssigner
    ) -> None:
        super().__init__()
        self._sequence = itertools.count()
        self._priority_queue: asyncio.PriorityQueue[
            DefaultActivityRequestScheduler.QueueActivityRequest
        ] = asyncio.PriorityQueue()
        self._priority_assigner = priority_assigner

    @override
    async def enqueue(self, request: ActivityRequest, future: Future[Any]) -> None:

        priority = self._priority_assigner.compute_priority_for_activity_request(
            request
        )
        enqueue_request = self.QueueActivityRequest(
            request=request,
            future=future,
            timestamp=time.monotonic_ns(),
            priority=priority,
            sequence=next(self._sequence),
        )

        await self._priority_queue.put(enqueue_request)

    @override
    async def dequeue(self) -> Tuple[ActivityRequest, Future[Any]]:
        queue_request = await self._priority_queue.get()

        return queue_request.request, queue_request.future

    @override
    async def length(self) -> int:
        return self._priority_queue.qsize()
