from abc import ABC, abstractmethod
from dataclasses import dataclass

from distributed_inference.application.inference.domain.inference_flow import (
    InferenceRequest,
)


class InferenceRequestScheduler(ABC):
    @dataclass(frozen=True)
    class _QueueInferenceRequest:
        inference_request: InferenceRequest
        enqueue_timestamp: float

    @abstractmethod
    async def enqueue(self, inference_request: InferenceRequest) -> None: ...

    @abstractmethod
    async def dequeue(self) -> InferenceRequest | None: ...

    @abstractmethod
    async def length(self) -> int: ...
