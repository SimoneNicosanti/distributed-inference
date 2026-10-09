from pydantic import BaseModel, ConfigDict

from shared.context.model_invocation_context import ModelInvocationContext
from shared.tensor.tensor_bundle import TensorBundle


class ModelInvocationResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    context: ModelInvocationContext
    bundle: TensorBundle
