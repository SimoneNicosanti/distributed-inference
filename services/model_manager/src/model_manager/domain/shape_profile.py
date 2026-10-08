from pydantic import BaseModel, ConfigDict, NonNegativeFloat, PositiveInt

from shared.model.keys import LayerKey, TensorKey


class InputShapePoint(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    shape: tuple[PositiveInt, ...]


## This is a dynamic shape configuration for which the profiling is done
## For each input, it tells a concrete shape it has
class ShapePoint(BaseModel):
    model_config = ConfigDict(frozen=True)

    input_shape_points: tuple[InputShapePoint, ...]


class LayerShapeProperty(BaseModel):
    model_config = ConfigDict(frozen=True)

    flops: NonNegativeFloat


class TensorShapeProperty(BaseModel):
    model_config = ConfigDict(frozen=True)

    shape: tuple[PositiveInt, ...]


class ShapeProfile(BaseModel):
    model_config = ConfigDict(frozen=True)

    layer_properties: dict[LayerKey, LayerShapeProperty]
    tensor_properties: dict[TensorKey, TensorShapeProperty]
