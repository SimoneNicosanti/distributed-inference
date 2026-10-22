from dataclasses import dataclass

from shared.context.model_pass_context import ModelPassContext, ModelPassId
from shared.tensor.tensor_bundle import TensorBundle


@dataclass(frozen=True, slots=True)
class ModelPassResultContribution:
    model_pass_context: ModelPassContext
    bundle: TensorBundle

    @property
    def model_pass_id(self) -> ModelPassId:
        return self.model_pass_context.model_pass_id
