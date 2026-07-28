from abc import ABC, abstractmethod

from distributed_inference.application.worker.contracts.execution.forwarding.route_instruction import (
    RouteInstruction,
)
from distributed_inference.application.worker.domain.inference_flow import (
    InferenceMessage,
)


class InferenceMessageForwarder(ABC):
    @abstractmethod
    def forward_inference_message(
        self, inference_message: InferenceMessage, route_instruction: RouteInstruction
    ) -> None: ...
