from abc import ABC, abstractmethod

from distributed_inference.worker.domain.inference_flow import (
    InferenceMessage,
)
from distributed_inference.worker.domain.route_instruction import (
    RouteInstruction,
)


class InferenceMessageForwarder(ABC):
    @abstractmethod
    def forward_inference_message(
        self, inference_message: InferenceMessage, route_instruction: RouteInstruction
    ) -> None: ...
