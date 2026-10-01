import asyncio
import itertools
import time
from asyncio import Future
from dataclasses import dataclass
from typing import Any, override

from shared.plan.plan import InferencePlanVersion, ServiceInferencePlan
from scheduling.queue_request import QueueRequest
from worker.application.deployment.contracts.service_inference_plan_preparer import (
    ServiceInferencePlanPreparer,
)
from worker.application.ports.outbound.service_inference_plan_store import (
    ServiceInferencePlanStore,
)
from worker.application.scheduling.contracts.sub_model_invocation_request_scheduler import (
    SubModelInvocationRequestScheduler,
)
from worker.domain.sub_model.invocation.sub_model_invocation_request_response import (
    SubModelInvocationRequest,
)


@dataclass
class QueueSubModelInvocationRequest(QueueRequest[SubModelInvocationRequest, Any]):
    plan_version: InferencePlanVersion
    per_plan_priority: int
    sequence: int

    def __lt__(self, other: Any) -> bool:
        if not isinstance(
            other,
            QueueSubModelInvocationRequest,
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


class PlanBasedSubModelInvocationRequestScheduler(
    SubModelInvocationRequestScheduler, ServiceInferencePlanPreparer
):
    def __init__(self, inference_plan_store: ServiceInferencePlanStore) -> None:
        self._lock = asyncio.Lock()

        self._priority_queue: asyncio.PriorityQueue[QueueSubModelInvocationRequest] = (
            asyncio.PriorityQueue()
        )

        self._sequence = itertools.count()

        self._plan_store = inference_plan_store

    @override
    async def enqueue(
        self, request: SubModelInvocationRequest, future: Future[Any]
    ) -> None:
        plan_version = request.plan_version

        async with self._lock:
            request_priority = await self._assign_priority(request)

            queue_request = QueueSubModelInvocationRequest(
                request=request,
                future=future,
                timestamp=time.monotonic_ns(),
                per_plan_priority=request_priority,
                sequence=next(self._sequence),
                plan_version=plan_version,
            )

            ## We do not need to wait because the enqueue is protected by the lock
            self._priority_queue.put_nowait(queue_request)

    @override
    async def dequeue(self) -> tuple[SubModelInvocationRequest, Future[Any]]:
        queue_request = await self._priority_queue.get()
        return queue_request.request, queue_request.future

    @override
    async def length(self) -> int:
        return self._priority_queue.qsize()

    @override
    async def prepare_service_inference_plan(
        self, service_inference_plan: ServiceInferencePlan
    ) -> None:
        ## We do not need special ops, we just need to sync with the plan in the store
        pass

    async def _assign_priority(self, request: SubModelInvocationRequest) -> int:
        plan_version = request.plan_version
        plan = await self._plan_store.get_service_inference_plan_by_version(
            plan_version
        )
        if plan is None:
            raise ValueError(f"Plan version {plan_version} does not exist")

        flow_id = request.flow_id
        sub_model_deployment = request.context.sub_model_deployment_id
        priority_value = plan.get_priority(flow_id, sub_model_deployment)
        if priority_value is None:
            raise ValueError(f"No priority for {flow_id}/{sub_model_deployment}")

        return priority_value
