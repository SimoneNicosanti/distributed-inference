from abc import ABC, abstractmethod
from typing import List

from distributed_inference.application.inference.contracts.execution.forwarding.route_instruction import (
    RouteInstruction,
)
from distributed_inference.application.inference.domain.inference_flow import (
    InferenceResponse,
)


class InferenceResponseRouter(ABC):
    @abstractmethod
    async def route_inference_response(
        self, inference_response: InferenceResponse
    ) -> List[RouteInstruction]: ...
