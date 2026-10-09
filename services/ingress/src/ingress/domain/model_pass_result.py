from pydantic import BaseModel, ConfigDict

from shared.context.model_pass_context import ModelPassContext
from shared.tensor.tensor_bundle import TensorBundle


class ModelPassResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    context: ModelPassContext
    bundle: TensorBundle
