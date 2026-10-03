from pydantic import BaseModel, ConfigDict, Field

from shared.artifact.artifact_ref import ArtifactRef
from shared.model.keys import LayerKey, TensorKey
from shared.model.model_variant import (
    AccuracyMetric,
    DynamicShape,
    ModelVariantFormat,
    ModelVariantId,
    NumericPrecision,
    QuantizationType,
    StaticShape,
)


class RedisDto(BaseModel):
    model_config = ConfigDict(frozen=True)


class InfoDto(RedisDto):
    precision: NumericPrecision
    quantization: QuantizationType
    accuracies: list[AccuracyMetric]
    format: ModelVariantFormat

    static_shapes: list[StaticShape]
    dynamic_shapes: list[DynamicShape]


class LayerDescriptionDto(RedisDto):
    name: str
    type: str
    is_input: bool
    is_output: bool


class EdgeDescriptionDto(RedisDto):
    source: LayerKey
    target: LayerKey
    tensors: set[TensorKey] = Field(default_factory=set)


class TopologyDto(RedisDto):
    layers: dict[LayerKey, LayerDescriptionDto]
    edges: list[EdgeDescriptionDto]


class WeightPropertyDto(RedisDto):
    data_type: str
    shape: tuple[int, ...]
    data_type_size_in_bytes: int


class LayerPropertyDto(RedisDto):
    weight_properties: dict[TensorKey, WeightPropertyDto]


class TensorPropertyDto(RedisDto):
    data_type: str
    data_type_size_in_bytes: int
    shape_expression: tuple[int | str, ...]


class ShapePointDto(RedisDto):
    dims: tuple[tuple[str, int], ...]


class LayerShapePropertyDto(RedisDto):
    flops: float


class TensorShapePropertyDto(RedisDto):
    shape: tuple[int, ...]


class ShapeProfileDto(RedisDto):
    shape_point: ShapePointDto
    layer_properties: dict[LayerKey, LayerShapePropertyDto]
    tensor_properties: dict[TensorKey, TensorShapePropertyDto]


class ProfileDto(RedisDto):
    topology: TopologyDto
    invariant_layer_properties: dict[LayerKey, LayerPropertyDto]
    invariant_tensor_properties: dict[TensorKey, TensorPropertyDto]
    shape_profiles: list[ShapeProfileDto]
    optimization_contractions: list[tuple[LayerKey, ...]]


class ProfiledModelVariantDto(RedisDto):
    model_variant_id: ModelVariantId
    info: InfoDto
    profile: ProfileDto
    artifact_ref: ArtifactRef
