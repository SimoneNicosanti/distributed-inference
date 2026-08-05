from enum import StrEnum, auto
from typing import Annotated, List, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from distributed_inference.model_manager.domain.model import ModelId, ModelType
from distributed_inference.model_manager.domain.model_version_graph import (
    ModelVersionGraph,
)


class ModelVersionPrecision(StrEnum):
    FP32 = auto()
    FP16 = auto()
    INT8 = auto()


class ModelVersionQuantization(StrEnum):
    NONE = auto()
    STATIC = auto()
    DYNAMIC = auto()


class AccuracyMetric(StrEnum):
    MAE = auto()
    RMSE = auto()


class ModelVersionFormat(StrEnum):
    ONNX = auto()
    TORCHSCRIPT = auto()


class ShapeType(StrEnum):
    BATCH = auto()
    SEQUENCE = auto()


class CNNArchitectureInfo(BaseModel):
    kind: Literal[ModelType.CNN] = ModelType.CNN


class TransformerArchitectureInfo(BaseModel):
    kind: Literal[ModelType.VIT, ModelType.BERT]
    num_heads: int
    hidden_size: int


class VITArchitectureInfo(TransformerArchitectureInfo):
    kind: Literal[ModelType.VIT] = ModelType.VIT


class BERTArchitectureInfo(TransformerArchitectureInfo):
    kind: Literal[ModelType.BERT] = ModelType.BERT


# class LLMArchitectureInfo(BaseModel):
#     kind: Literal[ModelType.LLM] = ModelType.LLM
#     context_length: int
#     vocab_size: int
#     num_kv_heads: int
#     # etc, grows here only, nowhere else touched


type ArchitectureInfo = Annotated[
    CNNArchitectureInfo | VITArchitectureInfo | BERTArchitectureInfo,
    Field(discriminator="kind"),
]


class ShapeInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: ShapeType
    name: str


class DynamicShapeInfo(ShapeInfo):
    min_value: int
    max_value: int
    step_size: int = 1

    @model_validator(mode="after")
    def validate_model(self) -> Self:

        if self.min_value <= 0:
            raise ValueError("min must be > 0")
        if self.max_value <= 0:
            raise ValueError("max must be > 0")
        if self.step_size <= 0:
            raise ValueError("steps must be > 0")

        if self.min_value >= self.max_value:
            raise ValueError("min must be < max")

        return self


class StaticShapeInfo(ShapeInfo):
    model_config = ConfigDict(frozen=True)

    value: int

    @model_validator(mode="after")
    def validate_model(self) -> Self:

        if self.value <= 0:
            raise ValueError("value must be > 0")

        return self


class ModelVersionInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    precision: ModelVersionPrecision
    quantization: ModelVersionQuantization
    accuracy: dict[AccuracyMetric, float]
    format: ModelVersionFormat

    static_shapes: List[StaticShapeInfo]
    dynamic_shapes: List[DynamicShapeInfo]

    architecture_info: ArchitectureInfo

    ## TODO: Add check for shapes: no shape can be declared twice in the same or different shape group
    ## TODO: Add check for coherence between model type and architecture info


class ModelVersionId(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_id: ModelId
    version_tag: str


class ModelVersion(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_version_id: ModelVersionId
    model_version_info: ModelVersionInfo

    @property
    def model_id(self) -> ModelId:
        return self.model_version_id.model_id


class ProfiledModelVersion(ModelVersion):
    model_version_graph: ModelVersionGraph
