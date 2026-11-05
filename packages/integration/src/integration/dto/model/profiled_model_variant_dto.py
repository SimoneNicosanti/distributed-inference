from typing import Annotated

from pydantic import (
    BaseModel,
    Field,
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


class InputShapeInvariableTensorPropertiesDto(BaseModel):
    data_type: str
    data_type_size_in_bytes: Annotated[PositiveInt, Field(strict=True)]
    shape_expression: tuple[
        Annotated[PositiveInt, Field(strict=True)]
        | Annotated[str, Field(min_length=1)],
        ...,
    ]


class InputShapeInvariableLayerPropertiesDto(BaseModel):
    weight_properties: dict[TensorKey, InputShapeInvariableTensorPropertiesDto]


class InputShapeInvariablePropertiesDto(BaseModel):
    layer_properties: dict[LayerKey, InputShapeInvariableLayerPropertiesDto]
    tensor_properties: dict[TensorKey, InputShapeInvariableTensorPropertiesDto]


class ModelInputConcreteShapesDto(BaseModel):
    input_shapes: dict[
        TensorKey, tuple[Annotated[PositiveInt, Field(strict=True)], ...]
    ]


class InputShapeVariableLayerPropertiesDto(BaseModel):
    flops: float = Field(strict=True, ge=0, allow_inf_nan=False)


class InputShapeVariableTensorPropertiesDto(BaseModel):
    shape: tuple[Annotated[PositiveInt, Field(strict=True)], ...]


class InputShapeVariableModelPropertiesDto(BaseModel):
    max_cut_size_mib: float = Field(
        strict=True,
        gt=0,
        allow_inf_nan=False,
        description="Maximum cut tensor volume in MiB (1 MiB = 1024 ** 2 bytes)",
    )


class InputShapeVariablePropertiesDto(BaseModel):
    input_shapes: ModelInputConcreteShapesDto
    layer_properties: dict[LayerKey, InputShapeVariableLayerPropertiesDto]
    tensor_properties: dict[TensorKey, InputShapeVariableTensorPropertiesDto]
    model_properties: InputShapeVariableModelPropertiesDto


class ProfileDto(BaseModel):
    topology: TopologyDto
    invariable_properties: InputShapeInvariablePropertiesDto
    variable_properties: list[InputShapeVariablePropertiesDto]
    optimization_contractions: list[tuple[LayerKey, ...]]

    @property
    def input_shape_configurations(self) -> list[ModelInputConcreteShapesDto]:
        return [properties.input_shapes for properties in self.variable_properties]


class ProfiledModelVariantDto(BaseModel):
    model_variant_id: ModelVariantId
    info: ModelVariantInfoDto
    profile: ProfileDto
    artifact_ref: ArtifactRef
