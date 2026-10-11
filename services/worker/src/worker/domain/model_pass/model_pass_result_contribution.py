from pydantic import BaseModel, ConfigDict

from shared.context.model_pass_context import ModelPassContext, ModelPassId
from shared.tensor.tensor_bundle import TensorBundle


class ModelPassResultContribution(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_pass_context: ModelPassContext
    bundle: TensorBundle

    @property
    def model_pass_id(self) -> ModelPassId:
        return self.model_pass_context.model_pass_id
