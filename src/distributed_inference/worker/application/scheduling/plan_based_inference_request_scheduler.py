import asyncio
import itertools
import time
from asyncio import Future
from dataclasses import dataclass
from typing import Any, override

from distributed_inference.building_blocks.scheduling.request_scheduler import (
    RequestScheduler,
)
from distributed_inference.domain.plan import InferencePlanVersion, ServiceInferencePlan
from distributed_inference.worker.application.deployment.contracts.inference_plan_preparer import (
    InferencePlanPreparer,
)
from distributed_inference.worker.application.scheduling.contracts.inference_request_scheduler import (
    InferenceRequestScheduler,
)
from distributed_inference.worker.application.scheduling.contracts.inference_request_static_priority_assigner import (
    InferenceRequestStaticPriorityAssigner,
)
from distributed_inference.worker.domain.inference_flow import (
    InferenceRequest,
)


class PlanBasedInferenceRequestScheduler(
    InferenceRequestScheduler, InferencePlanPreparer
):
    @dataclass
    class QueueInferenceRequest(RequestScheduler.QueueRequest):
        plan_version: InferencePlanVersion
        per_plan_priority: int
        sequence: int

        def __lt__(self, other: Any) -> bool:
            if not isinstance(
                other, PlanBasedInferenceRequestScheduler.QueueInferenceRequest
            ):
                return NotImplemented

            return (
                self.plan_version,
                self.per_plan_priority,
                self.timestamp,
                self.sequence,
            ) < (
                other.plan_version,
                other.per_plan_priority,
                other.timestamp,
                other.sequence,
            )

    def __init__(self) -> None:
        self._lock = asyncio.Lock()

        self._priority_queue: asyncio.PriorityQueue[
            PlanBasedInferenceRequestScheduler.QueueInferenceRequest
        ] = asyncio.PriorityQueue()

        self._sequence = itertools.count()

        self._priority_assigners: dict[
            InferencePlanVersion, InferenceRequestStaticPriorityAssigner
        ] = {}

    @override
    async def enqueue(self, request: InferenceRequest, future: Future[Any]) -> None:
        plan_version = request.inference_plan_version

        async with self._lock:
            priority_assigner = self._priority_assigners.get(plan_version, None)
            if priority_assigner is None:
                raise ValueError(f"Plan version {plan_version} does not exist")
            per_plan_priority = priority_assigner.assign_priority(request)

            queue_request = self.QueueInferenceRequest(
                request=request,
                future=future,
                timestamp=time.monotonic_ns(),
                per_plan_priority=per_plan_priority,
                sequence=next(self._sequence),
                plan_version=plan_version,
            )

            ## We do not need to wait because the enqueue is protected by the lock
            self._priority_queue.put_nowait(queue_request)

    @override
    async def dequeue(self) -> tuple[InferenceRequest, Future[Any]]:
        queue_request = await self._priority_queue.get()
        return queue_request.request, queue_request.future

    @override
    async def length(self) -> int:
        return self._priority_queue.qsize()

    @override
    async def prepare_inference_plan(
        self, service_inference_plan: ServiceInferencePlan
    ) -> None:
        async with self._lock:
            ## We make this implement this interface so it can adapt to the plan when needed
            ## We do not need the inference plan store: when a new plan is received, we just
            ## need to create a new priority assigner, no plan prepare sync is needed
            raise NotImplementedError
