from abc import ABC, abstractmethod
from asyncio import Future
from typing import Any, Tuple, override

from distributed_inference.building_blocks.scheduling.request_scheduler import (
    RequestScheduler,
)
from distributed_inference.worker.domain.sub_model_inference_request_response import (
    SubModelInferenceRequest,
)


class SubModelInferenceRequestScheduler(RequestScheduler, ABC):
    class QueueSubModelInferenceRequest(RequestScheduler.QueueRequest):
        pass

    @abstractmethod
    @override
    async def enqueue(
        self, request: SubModelInferenceRequest, future: Future[Any]
    ) -> None: ...

    @abstractmethod
    @override
    async def dequeue(self) -> Tuple[SubModelInferenceRequest, Future[Any]]: ...

    @abstractmethod
    @override
    async def length(self) -> int: ...
