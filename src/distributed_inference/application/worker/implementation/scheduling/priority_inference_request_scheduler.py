import time
from collections import deque
from dataclasses import dataclass
from typing import Any, override

import aiorwlock

from distributed_inference.application.worker.contracts.execution.scheduling.inference_request_scheduler import (
    InferenceRequestScheduler,
)
from distributed_inference.application.worker.domain.inference_flow import (
    InferenceRequest,
)


class PriorityInferenceRequestScheduler(InferenceRequestScheduler):
    @dataclass(frozen=True)
    class _QueueInferenceRequest(InferenceRequestScheduler._QueueInferenceRequest):
        priority: int

    def __init__(self, priorities: int, plan: Any) -> None:
        self._lock = aiorwlock.RWLock()
        self._plan = plan
        self._priority_queues: list[
            deque[PriorityInferenceRequestScheduler._QueueInferenceRequest]
        ] = [deque() for _ in range(priorities)]

    @override
    async def enqueue(self, inference_request: InferenceRequest) -> None:
        ## TODO: decide request priority based on plan
        request_priority = 0

        async with self._lock.writer_lock:
            queue_inference_request = self._QueueInferenceRequest(
                inference_request=inference_request,
                enqueue_timestamp=time.monotonic_ns(),
                priority=request_priority,
            )

            self._priority_queues[request_priority].appendleft(queue_inference_request)

    @override
    async def dequeue(self) -> InferenceRequest | None:
        async with self._lock.writer_lock:
            for priority_queue in self._priority_queues:
                if len(priority_queue) > 0:
                    next_queue_inference_request = priority_queue.pop()
                    return next_queue_inference_request.inference_request
        return None

    @override
    async def length(self) -> int:
        total_len = 0
        async with self._lock.reader_lock:
            for priority_queue in self._priority_queues:
                total_len += len(priority_queue)
        return total_len

    async def length_from_priority(self, priority: int) -> int:
        if priority >= len(self._priority_queues):
            raise ValueError(f"Priority {priority} does not exist")
        async with self._lock.reader_lock:
            return len(self._priority_queues[priority])
