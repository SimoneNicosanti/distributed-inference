from abc import ABC, abstractmethod

from distributed_inference.application.inference.domain.inference_request import (
    InferenceRequest,
)
from distributed_inference.application.inference.domain.inference_response import (
    InferenceResponse,
)


## This is the inference worker for a single sub-model
## It handles only one inference
class InferenceWorker(ABC):
    @abstractmethod
    def run_inference_request(self, request: InferenceRequest) -> InferenceResponse: ...
