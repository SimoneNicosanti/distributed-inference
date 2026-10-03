from pydantic import BaseModel, ConfigDict

from shared.model.keys import TensorKey


class WeightProperty(BaseModel):
    model_config = ConfigDict(frozen=True)

    data_type: str
    shape: tuple[int, ...]
    data_type_size_in_bytes: int


class LayerProperty(BaseModel):
    model_config = ConfigDict(frozen=True)

    weight_properties: dict[TensorKey, WeightProperty]
