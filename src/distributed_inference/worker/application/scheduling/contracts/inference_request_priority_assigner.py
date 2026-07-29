from abc import ABC, abstractmethod

from distributed_inference.worker.domain.inference_flow import (
    InferenceRequest,
)


class InferenceRequestPriorityAssigner(ABC):
    @abstractmethod
    def compute_priority(self, inference_request: InferenceRequest) -> int: ...
