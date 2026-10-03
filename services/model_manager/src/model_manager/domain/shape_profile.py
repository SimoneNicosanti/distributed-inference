from pydantic import BaseModel, ConfigDict

from shared.model.keys import LayerKey, TensorKey


## This is a dynamic shape configuration for which the profiling is done
## It wraps a tuple of tuple like:
## ((batch_size, x), (sequence_size, y))
## Multiple ShapePoints are the different configurations we are profiling
class ShapePoint(BaseModel):
    model_config = ConfigDict(frozen=True)

    dims: tuple[tuple[str, int], ...]


class LayerShapeProperty(BaseModel):
    model_config = ConfigDict(frozen=True)

    flops: float


class TensorShapeProperty(BaseModel):
    model_config = ConfigDict(frozen=True)

    shape: tuple[int, ...]


class ShapeProfile(BaseModel):
    model_config = ConfigDict(frozen=True)

    layer_properties: dict[LayerKey, LayerShapeProperty]
    tensor_properties: dict[TensorKey, TensorShapeProperty]
