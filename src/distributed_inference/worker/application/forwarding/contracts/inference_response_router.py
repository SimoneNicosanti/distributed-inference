from abc import ABC, abstractmethod
from typing import List

from distributed_inference.worker.domain.inference_flow import (
    InferenceResponse,
)
from distributed_inference.worker.domain.route_instruction import (
    RouteInstruction,
)


class InferenceResponseRouter(ABC):
    @abstractmethod
    async def route_inference_response(
        self, inference_response: InferenceResponse
    ) -> List[RouteInstruction]: ...
