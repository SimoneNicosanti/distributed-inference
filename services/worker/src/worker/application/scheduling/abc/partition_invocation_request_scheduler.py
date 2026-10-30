from abc import ABC, abstractmethod
from asyncio import Future
from typing import Any, override

from scheduling.request_scheduler import RequestScheduler
from worker.domain.hop.hop import ExecutableHop
from worker.domain.hop.hop_invocation import (
    HopInvocationRequest,
)


class HopInvocationRequestScheduler(
    RequestScheduler[HopInvocationRequest[ExecutableHop], Any], ABC
):
    @abstractmethod
    @override
    async def enqueue(
        self, request: HopInvocationRequest[ExecutableHop], future: Future[Any]
    ) -> None: ...

    @abstractmethod
    @override
    async def dequeue(
        self,
    ) -> tuple[HopInvocationRequest[ExecutableHop], Future[Any]]: ...

    @abstractmethod
    @override
    async def length(self) -> int: ...
