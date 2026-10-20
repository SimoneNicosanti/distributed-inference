from pydantic import BaseModel, ConfigDict, NonNegativeFloat, PositiveInt

from controller.domain.profile.model_variant_topology import ModelVariantTopology
from shared.model.keys import LayerKey, TensorKey
from shared.model.model import ModelTask, ModelVisibility
from shared.model.model_variant import ModelVariantId, PerformanceMetric


class ModelVariantInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    visibility: ModelVisibility
    task: ModelTask

    accuracies: tuple[PerformanceMetric, ...]


class WeightProperty(BaseModel):
    model_config = ConfigDict(frozen=True)

    data_type: str
    shape: tuple[int, ...]
    data_type_size_in_bytes: int


class LayerProperty(BaseModel):
    model_config = ConfigDict(frozen=True)

    weight_properties: dict[TensorKey, WeightProperty]


class TensorProperty(BaseModel):
    model_config = ConfigDict(frozen=True)

    data_type: str
    data_type_size_in_bytes: int
    shape_expression: tuple[int | str, ...]


class InputShapePoint(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    shape: tuple[PositiveInt, ...]


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

    shape_point: ShapePoint
    layer_properties: dict[LayerKey, LayerShapeProperty]
    tensor_properties: dict[TensorKey, TensorShapeProperty]


class ModelVariantStaticProfile(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    model_variant_id: ModelVariantId

    info: ModelVariantInfo
    topology: ModelVariantTopology
    invariant_layer_properties: dict[LayerKey, LayerProperty]
    invariant_tensor_properties: dict[TensorKey, TensorProperty]
    shape_profiles: tuple[ShapeProfile, ...]
    optimization_contractions: tuple[tuple[LayerKey, ...], ...]
