from pydantic import BaseModel, ConfigDict

from shared.context.model_pass_context import ModelPassContext
from shared.tensor.tensor_bundle import TensorBundle


class ModelPassResultContribution(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_pass_context: ModelPassContext
    bundle: TensorBundle
