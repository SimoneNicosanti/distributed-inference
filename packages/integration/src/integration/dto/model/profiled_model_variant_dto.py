from pydantic import (
    BaseModel,
    Field,
    NonNegativeFloat,
    PositiveInt,
)

from integration.dto.model.model_variant_dto import ModelVariantInfoDto
from shared.artifact.artifact_ref import ArtifactRef
from shared.model.keys import LayerKey, TensorKey
from shared.model.model_variant import (
    ModelVariantId,
)


class LayerDescriptionDto(BaseModel):
    name: str
    type: str
    is_input: bool
    is_output: bool


class EdgeDescriptionDto(BaseModel):
    source: LayerKey
    target: LayerKey
    tensors: set[TensorKey] = Field(default_factory=set)


class TopologyDto(BaseModel):
    layers: dict[LayerKey, LayerDescriptionDto]
    edges: list[EdgeDescriptionDto]


class WeightPropertyDto(BaseModel):
    data_type: str
    shape: tuple[int, ...]
    data_type_size_in_bytes: int


class LayerPropertyDto(BaseModel):
    weight_properties: dict[TensorKey, WeightPropertyDto]


class TensorPropertyDto(BaseModel):
    data_type: str
    data_type_size_in_bytes: int
    shape_expression: tuple[int | str, ...]


class InputShapePointDto(BaseModel):
    name: str
    shape: tuple[PositiveInt, ...]


class ShapePointDto(BaseModel):
    input_shape_points: tuple[InputShapePointDto, ...]


class LayerShapePropertyDto(BaseModel):
    flops: NonNegativeFloat


class TensorShapePropertyDto(BaseModel):
    shape: tuple[PositiveInt, ...]


class ShapeProfileDto(BaseModel):
    shape_point: ShapePointDto
    layer_properties: dict[LayerKey, LayerShapePropertyDto]
    tensor_properties: dict[TensorKey, TensorShapePropertyDto]


class ProfileDto(BaseModel):
    topology: TopologyDto
    invariant_layer_properties: dict[LayerKey, LayerPropertyDto]
    invariant_tensor_properties: dict[TensorKey, TensorPropertyDto]
    shape_profiles: list[ShapeProfileDto]
    optimization_contractions: list[tuple[LayerKey, ...]]

    @property
    def shape_points(self) -> list[ShapePointDto]:
        return [shape_profile.shape_point for shape_profile in self.shape_profiles]


class ProfiledModelVariantDto(BaseModel):
    model_variant_id: ModelVariantId
    info: ModelVariantInfoDto
    profile: ProfileDto
    artifact_ref: ArtifactRef
