from abc import ABC, abstractmethod

from distributed_inference.worker.domain.sub_model_inference_message import (
    SubModelInferenceMessage,
)
from distributed_inference.worker.domain.sub_model_output_route_instruction import (
    SubModelOutputRouteInstruction,
)


class SubModelInferenceMessageForwarder(ABC):
    @abstractmethod
    def forward_sub_model_inference_message(
        self,
        sub_model_inference_message: SubModelInferenceMessage,
        sub_model_output_route_instruction: SubModelOutputRouteInstruction,
    ) -> None: ...
