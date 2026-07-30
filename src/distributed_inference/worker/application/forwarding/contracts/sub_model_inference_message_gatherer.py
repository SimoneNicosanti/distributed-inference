from abc import ABC, abstractmethod

from distributed_inference.worker.domain.sub_model_inference_message import (
    SubModelInferenceMessage,
)
from distributed_inference.worker.domain.sub_model_inference_request_response import (
    SubModelInferenceRequest,
)


class SubModelInferenceMessageGatherer(ABC):
    @abstractmethod
    async def gather_sub_model_inference_message(
        self, sub_model_inference_message: SubModelInferenceMessage
    ) -> SubModelInferenceRequest | None: ...
