import asyncio
import itertools
import time
from asyncio import Future
from dataclasses import dataclass
from typing import Any, override

from scheduling.queue_request import QueueRequest
from shared.plan.plan_version import PlanVersion
from worker.application.ports.outbound.plan_store.worker_sub_plan_readers import (
    SchedulingPlanReader,
)
from worker.application.scheduling.abc.partition_invocation_request_scheduler import (
    HopInvocationRequestScheduler,
)
from worker.domain.hop.hop import ExecutableHop
from worker.domain.hop.hop_invocation import (
    HopInvocationRequest,
)
from worker.domain.plan.scheduling_plan import PriorityKey


@dataclass
class QueuedPartitionInvocationRequest(QueueRequest[HopInvocationRequest, Any]):
    plan_version: PlanVersion
    per_plan_priority: int
    sequence: int

    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, QueuedPartitionInvocationRequest):
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


class PlanBasedPartitionInvocationRequestScheduler(HopInvocationRequestScheduler):
    def __init__(self, scheduling_plan_reader: SchedulingPlanReader) -> None:
        self._lock = asyncio.Lock()
        self._priority_queue: asyncio.PriorityQueue[
            QueuedPartitionInvocationRequest
        ] = asyncio.PriorityQueue()
        self._sequence = itertools.count()
        self._scheduling_plan_reader = scheduling_plan_reader

    @override
    async def enqueue(
        self, request: HopInvocationRequest[ExecutableHop], future: Future[Any]
    ) -> None:
        async with self._lock:
            request_priority = await self._assign_priority(request)
            queued_request = QueuedPartitionInvocationRequest(
                request=request,
                future=future,
                timestamp=time.monotonic_ns(),
                per_plan_priority=request_priority,
                sequence=next(self._sequence),
                plan_version=request.plan_version,
            )
            self._priority_queue.put_nowait(queued_request)

    @override
    async def dequeue(self) -> tuple[HopInvocationRequest[ExecutableHop], Future[Any]]:
        queued_request = await self._priority_queue.get()
        return queued_request.request, queued_request.future

    @override
    async def length(self) -> int:
        return self._priority_queue.qsize()

    async def _assign_priority(
        self, request: HopInvocationRequest[ExecutableHop]
    ) -> int:
        plan_version = request.plan_version
        scheduling_plan = (
            await self._scheduling_plan_reader.get_scheduling_plan_by_version(
                plan_version
            )
        )
        priority_key = PriorityKey(
            flow_id=request.flow_id,
            executable_hop=request.hop,
        )
        return scheduling_plan.get_priority_by_key(priority_key)
