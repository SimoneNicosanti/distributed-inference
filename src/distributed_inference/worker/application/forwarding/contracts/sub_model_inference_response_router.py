from abc import ABC, abstractmethod
from typing import List

from distributed_inference.worker.domain.sub_model_inference_request_response import (
    SubModelInferenceResponse,
)
from distributed_inference.worker.domain.sub_model_output_route_instruction import (
    SubModelOutputRouteInstruction,
)


class SubModelInferenceResponseRouter(ABC):
    @abstractmethod
    async def route_sub_model_inference_response(
        self, sub_model_inference_response: SubModelInferenceResponse
    ) -> List[SubModelOutputRouteInstruction]: ...
