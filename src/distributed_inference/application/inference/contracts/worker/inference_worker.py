from abc import ABC, abstractmethod

from distributed_inference.application.inference.domain.inference_request import (
    InferenceRequest,
)
from distributed_inference.application.inference.domain.inference_response import (
    InferenceResponse,
)


class InferenceWorker(ABC):
    @abstractmethod
    def run_inference_request(self, request: InferenceRequest) -> InferenceResponse: ...
