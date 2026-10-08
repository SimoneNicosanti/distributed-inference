from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    PositiveInt,
)

from shared.artifact.artifact_ref import ArtifactRef
from shared.model.keys import LayerKey, TensorKey
from shared.model.model_variant import (
    AccuracyMetric,
    ModelVariantFormat,
    ModelVariantId,
    NumericPrecision,
    QuantizationType,
)


class RedisDto(BaseModel):
    model_config = ConfigDict(frozen=True)


class DynamicShapeValuesDto(RedisDto):
    values: frozenset[PositiveInt]


class InputInfoDto(RedisDto):
    shape: tuple[str | PositiveInt, ...]


class InfoDto(RedisDto):
    precision: NumericPrecision
    quantization: QuantizationType
    accuracies: list[AccuracyMetric]
    format: ModelVariantFormat

    inputs_info: dict[str, InputInfoDto]
    dynamic_dimension_values: dict[str, DynamicShapeValuesDto]


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


class InputShapePointDto(RedisDto):
    name: str
    shape: tuple[PositiveInt, ...]


class ShapePointDto(RedisDto):
    input_shape_points: tuple[InputShapePointDto, ...]


class LayerShapePropertyDto(RedisDto):
    flops: NonNegativeFloat


class TensorShapePropertyDto(RedisDto):
    shape: tuple[PositiveInt, ...]


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

    @property
    def shape_points(self) -> list[ShapePointDto]:
        return [shape_profile.shape_point for shape_profile in self.shape_profiles]


class ProfiledModelVariantDto(RedisDto):
    model_variant_id: ModelVariantId
    info: InfoDto
    profile: ProfileDto
    artifact_ref: ArtifactRef
