import asyncio
import time
from asyncio import Future
from collections import deque
from dataclasses import dataclass
from typing import Any, Tuple, override

from distributed_inference.application.scheduling.contracts.request_scheduler import (
    RequestScheduler,
)
from distributed_inference.application.worker.domain.inference_flow import (
    InferenceRequest,
)
from distributed_inference.domain.plan import ServiceInferencePlan


class PriorityInferenceRequestScheduler(RequestScheduler):
    @dataclass
    class _QueueInferenceRequest(RequestScheduler._QueueRequest):
        priority: int

    def __init__(self, priorities: int, plan: ServiceInferencePlan) -> None:
        self._condition = asyncio.Condition()
        self._plan = plan
        self._priority_queues: list[
            deque[PriorityInferenceRequestScheduler._QueueInferenceRequest]
        ] = [deque() for _ in range(priorities)]

    @override
    async def enqueue(self, request: InferenceRequest, future: Future[Any]) -> None:
        request_priority = 0  ## TODO: decide request priority based on plan
        queue_inference_request = self._QueueInferenceRequest(
            request=request,
            future=future,
            enqueue_timestamp=time.monotonic_ns(),
            priority=request_priority,
        )

        async with self._condition:  ## Takes the lock on the condition
            self._priority_queues[request_priority].append(queue_inference_request)

            ## Notifies that the condition state has changed
            ## If the coroutines where waiting for different conditions, notify all might have been better
            self._condition.notify(n=1)

    @override
    async def dequeue(self) -> Tuple[InferenceRequest, Future[Any]]:
        async with self._condition:  ## Takes the lock on the condition
            ## Waits until the condition state has changed
            ## In the meantime, the lock is released
            ## Once the condition state has changed, the lock is reacquired
            ## We declare the type of event we are waiting for
            await self._condition.wait_for(lambda: any(self._priority_queues))
            for priority_queue in self._priority_queues:
                if len(priority_queue) > 0:
                    next_queue_inference_request = priority_queue.popleft()
                    inference_request = next_queue_inference_request.request
                    future = next_queue_inference_request.future
                    return inference_request, future

        raise RuntimeError("Unreachable non empty queue")

    @override
    async def length(self) -> int:
        async with self._condition:
            return sum(len(queue) for queue in self._priority_queues)
