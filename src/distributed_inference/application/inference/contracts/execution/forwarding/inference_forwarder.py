from abc import ABC, abstractmethod

from distributed_inference.application.inference.domain.inference_request import (
    InferenceRequest,
)


class InferenceForwarder(ABC):
    @abstractmethod
    def forward_inference(self, inference_request: InferenceRequest) -> None: ...
