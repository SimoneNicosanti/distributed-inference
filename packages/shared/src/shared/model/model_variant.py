from enum import StrEnum, auto
from typing import Self

from pydantic import BaseModel, ConfigDict, PositiveInt, model_validator

from shared.model.model import (
    ModelId,
)


class NumericPrecision(StrEnum):
    FP32 = auto()
    FP16 = auto()
    INT8 = auto()


class QuantizationType(StrEnum):
    NONE = auto()
    STATIC = auto()
    DYNAMIC = auto()


class AccuracyMetricType(StrEnum):
    MAE = auto()
    RMSE = auto()


class AccuracyMetric(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: AccuracyMetricType
    dataset: str
    value: float


class ModelVariantFormat(StrEnum):
    ONNX = auto()
    TORCH_EXPORTED_PROGRAM = auto()


class ShapeType(StrEnum):
    BATCH = auto()
    SEQUENCE = auto()


# class LLMArchitectureInfo(BaseModel):
#     kind: Literal[ModelType.LLM] = ModelType.LLM
#     context_length: int
#     vocab_size: int
#     num_kv_heads: int
#     # etc, grows here only, nowhere else touched


class Shape(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: ShapeType
    name: str


class DynamicShape(Shape):
    min_value: PositiveInt
    max_value: PositiveInt
    step_size: PositiveInt = 1

    @model_validator(mode="after")
    def validate_model(self) -> Self:
        if self.min_value >= self.max_value:
            raise ValueError("min must be < max")

        return self


class StaticShape(Shape):
    model_config = ConfigDict(frozen=True)

    value: PositiveInt


class StateTensorKind(StrEnum):
    SELF_ATTENTION = auto()
    CROSS_ATTENTION = auto()


class StateTensorInfo(BaseModel):
    model_config = ConfigDict(frozen=True)
    input_name: str
    output_name: str
    kind: StateTensorKind


class ModelVariantId(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_id: ModelId
    variant_tag: str
