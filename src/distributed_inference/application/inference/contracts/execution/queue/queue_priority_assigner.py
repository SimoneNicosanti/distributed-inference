from abc import ABC, abstractmethod

from distributed_inference.application.inference.domain.inference_request import (
    InferenceRequest,
)


class QueuePriorityAssigner(ABC):
    @abstractmethod
    def assign_priority(self, request: InferenceRequest) -> int: ...
