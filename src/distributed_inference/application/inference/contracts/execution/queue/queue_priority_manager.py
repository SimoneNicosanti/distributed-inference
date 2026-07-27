from abc import ABC, abstractmethod

from distributed_inference.application.inference.domain.inference_request import (
    InferenceRequest,
)


class QueuePriorityManager(ABC):
    @abstractmethod
    def enque_with_priority(self, item: InferenceRequest, priority: int) -> None: ...

    @abstractmethod
    def deque(self) -> InferenceRequest | None: ...

    @abstractmethod
    def deque_from_priority(self, priority: int) -> InferenceRequest | None: ...

    @abstractmethod
    def length(self) -> int: ...

    @abstractmethod
    def length_from_priority(self, priority: int) -> int: ...
