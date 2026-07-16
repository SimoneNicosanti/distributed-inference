

from pydantic import BaseModel, ConfigDict


INPUT_LAYER_NAME = "InputLayer"
OUTPUT_LAYER_NAME = "OutputLayer"


class LayerInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str

    type: str

    flops: float
    weights_size: float

    inputs: dict[str, float]

    outputs: dict[str, float]

    is_input: bool
    is_output: bool


class EdgeInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    source: str
    target: str

    tensors: dict[str, float]
