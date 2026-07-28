from abc import ABC, abstractmethod
from typing import List

from distributed_inference.application.worker.contracts.execution.forwarding.route_instruction import (
    RouteInstruction,
)
from distributed_inference.application.worker.domain.inference_flow import (
    InferenceResponse,
)


class InferenceResponseRouter(ABC):
    @abstractmethod
    async def route_inference_response(
        self, inference_response: InferenceResponse
    ) -> List[RouteInstruction]: ...
