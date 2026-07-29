from abc import ABC, abstractmethod
from asyncio import Future
from typing import Any, Tuple, override

from distributed_inference.application.scheduling.contracts.request_scheduler import (
    RequestScheduler,
)
from distributed_inference.application.worker.domain.inference_flow import (
    InferenceRequest,
)


class InferenceRequestScheduler(RequestScheduler, ABC):
    class QueueInferenceRequest(RequestScheduler.QueueRequest):
        pass

    @abstractmethod
    @override
    async def enqueue(self, request: InferenceRequest, future: Future[Any]) -> None: ...

    @abstractmethod
    @override
    async def dequeue(self) -> Tuple[InferenceRequest, Future[Any]]: ...

    @abstractmethod
    @override
    async def length(self) -> int: ...
