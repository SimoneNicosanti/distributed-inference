from abc import ABC, abstractmethod

from distributed_inference.worker.domain.sub_model_inference_request_response import (
    SubModelInferenceRequest,
    SubModelInferenceResponse,
)


## This is the inference coordinator for a pool of inference workers
class SubModelExecutionCoordinator(ABC):
    @abstractmethod
    async def process_sub_model_inference_request(
        self, sub_model_inference_request: SubModelInferenceRequest
    ) -> SubModelInferenceResponse: ...
