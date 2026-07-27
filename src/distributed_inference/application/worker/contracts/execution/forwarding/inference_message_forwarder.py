from abc import ABC, abstractmethod

from distributed_inference.application.inference.contracts.execution.forwarding.route_instruction import (
    RouteInstruction,
)
from distributed_inference.application.inference.domain.inference_flow import (
    InferenceMessage,
)


class InferenceForwarder(ABC):
    @abstractmethod
    def forward_inference_message(
        self, inference_message: InferenceMessage, route_instruction: RouteInstruction
    ) -> None: ...
