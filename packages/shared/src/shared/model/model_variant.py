from enum import StrEnum, auto

from pydantic import BaseModel, ConfigDict

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


class PerformanceMetricType(StrEnum):
    MAE = auto()
    RMSE = auto()

    MAP_AT_50 = auto()
    MAP_AT_75 = auto()


class PerformanceMetric(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: PerformanceMetricType
    dataset: str
    value: float


class ModelVariantFormat(StrEnum):
    ONNX = auto()
    TORCH_EXPORTED_PROGRAM = auto()


# class LLMArchitectureInfo(BaseModel):
#     kind: Literal[ModelType.LLM] = ModelType.LLM
#     context_length: int
#     vocab_size: int
#     num_kv_heads: int
#     # etc, grows here only, nowhere else touched


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
