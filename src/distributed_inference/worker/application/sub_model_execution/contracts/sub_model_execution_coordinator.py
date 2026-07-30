from abc import ABC, abstractmethod

from distributed_inference.worker.domain.sub_model.invocation.sub_model_invocation_request_response import (
    SubModelInvocationRequest,
    SubModelInvocationResponse,
)


## This is the inference coordinator for a pool of inference workers
class SubModelExecutionCoordinator(ABC):
    @abstractmethod
    async def process_sub_model_invocation_request(
        self, sub_model_invocation_request: SubModelInvocationRequest
    ) -> SubModelInvocationResponse: ...
