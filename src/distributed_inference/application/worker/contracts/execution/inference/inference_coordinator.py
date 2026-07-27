from abc import ABC, abstractmethod

from distributed_inference.application.inference.domain.inference_flow import (
    InferenceRequest,
    InferenceResponse,
)


## This is the inference coordinator for a pool of inference workers
class InferenceCoordinator(ABC):
    @abstractmethod
    async def process_inference_request(
        self, inference_request: InferenceRequest
    ) -> InferenceResponse: ...
