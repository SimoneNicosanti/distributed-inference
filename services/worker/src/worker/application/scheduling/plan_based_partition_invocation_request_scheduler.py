import asyncio
import itertools
import time
from asyncio import Future
from dataclasses import dataclass
from typing import Any, override

from scheduling.queue_request import QueueRequest
from shared.plan.plan import InferencePlanVersion, ServiceInferencePlan
from worker.application.deployment.abc.service_inference_plan_preparer import (
    ServiceInferencePlanPreparer,
)
from worker.application.ports.outbound.service_inference_plan_store import (
    ServiceInferencePlanStore,
)
from worker.application.scheduling.abc.partition_invocation_request_scheduler import (
    PartitionInvocationRequestScheduler,
)
from worker.domain.partition.partition_invocation import (
    PartitionInvocationRequest,
)


@dataclass
class QueuedPartitionInvocationRequest(QueueRequest[PartitionInvocationRequest, Any]):
    plan_version: InferencePlanVersion
    per_plan_priority: int
    sequence: int

    def __lt__(self, other: Any) -> bool:
        if not isinstance(
            other,
            QueuedPartitionInvocationRequest,
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


class PlanBasedPartitionInvocationRequestScheduler(
    PartitionInvocationRequestScheduler, ServiceInferencePlanPreparer
):
    def __init__(self, inference_plan_store: ServiceInferencePlanStore) -> None:
        self._lock = asyncio.Lock()

        self._priority_queue: asyncio.PriorityQueue[
            QueuedPartitionInvocationRequest
        ] = asyncio.PriorityQueue()

        self._sequence = itertools.count()

        self._plan_store = inference_plan_store

    @override
    async def enqueue(
        self, request: PartitionInvocationRequest, future: Future[Any]
    ) -> None:
        plan_version = request.plan_version

        async with self._lock:
            request_priority = await self._assign_priority(request)

            queued_request = QueuedPartitionInvocationRequest(
                request=request,
                future=future,
                timestamp=time.monotonic_ns(),
                per_plan_priority=request_priority,
                sequence=next(self._sequence),
                plan_version=plan_version,
            )

            ## We do not need to wait because the enqueue is protected by the lock
            self._priority_queue.put_nowait(queued_request)

    @override
    async def dequeue(self) -> tuple[PartitionInvocationRequest, Future[Any]]:
        queued_request = await self._priority_queue.get()
        return queued_request.request, queued_request.future

    @override
    async def length(self) -> int:
        return self._priority_queue.qsize()

    @override
    async def prepare_service_inference_plan(
        self, service_inference_plan: ServiceInferencePlan
    ) -> None:
        ## We do not need special ops, we just need to sync with the plan in the store
        pass

    async def _assign_priority(self, request: PartitionInvocationRequest) -> int:
        plan_version = request.plan_version
        plan = await self._plan_store.get_service_inference_plan_by_version(
            plan_version
        )
        if plan is None:
            raise ValueError(f"Plan version {plan_version} does not exist")

        flow_id = request.flow_id
        partition_deployment = request.partition_deployment
        priority_value = plan.get_priority(flow_id, partition_deployment)
        if priority_value is None:
            raise ValueError(f"No priority for {flow_id}/{partition_deployment}")

        return priority_value
