from abc import ABC, abstractmethod

from distributed_inference.worker.domain.inference_flow import (
    InferenceRequest,
    InferenceResponse,
)


## This is the inference coordinator for a pool of inference workers
class InferenceWorkerCoordinator(ABC):
    @abstractmethod
    async def process_inference_request(
        self, inference_request: InferenceRequest
    ) -> InferenceResponse: ...
