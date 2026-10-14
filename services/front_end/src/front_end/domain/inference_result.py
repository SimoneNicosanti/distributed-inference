from pydantic import BaseModel, ConfigDict

from shared.tensor.tensor_bundle import TensorBundle


class InferenceResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    bundle: TensorBundle
