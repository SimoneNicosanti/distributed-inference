from abc import ABC, abstractmethod
from asyncio import Future
from typing import Any, override

from scheduling.request_scheduler import RequestScheduler
from shared.plan.hop import PlannedExecutableHopId
from worker.domain.hop.hop_invocation import (
    HopInvocationRequest,
)


class HopInvocationRequestScheduler(
    RequestScheduler[HopInvocationRequest[PlannedExecutableHopId], Any], ABC
):
    @abstractmethod
    @override
    async def enqueue(
        self, request: HopInvocationRequest[PlannedExecutableHopId], future: Future[Any]
    ) -> None: ...

    @abstractmethod
    @override
    async def dequeue(
        self,
    ) -> tuple[HopInvocationRequest[PlannedExecutableHopId], Future[Any]]: ...

    @abstractmethod
    @override
    async def length(self) -> int: ...
