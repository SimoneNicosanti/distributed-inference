from dataclasses import dataclass
from enum import StrEnum, auto

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


@dataclass(frozen=True, slots=True)
class PerformanceMetric:
    type: PerformanceMetricType
    dataset: str
    value: float


class ModelVariantFormat(StrEnum):
    ONNX = auto()
    TORCH_EXPORTED_PROGRAM = auto()


# @dataclass(frozen=True, slots=True, kw_only=True)
# class LLMArchitectureInfo:
#     kind: Literal[ModelType.LLM] = ModelType.LLM
#     context_length: int
#     vocab_size: int
#     num_kv_heads: int
#     # etc, grows here only, nowhere else touched


class StateTensorKind(StrEnum):
    SELF_ATTENTION = auto()
    CROSS_ATTENTION = auto()


@dataclass(frozen=True, slots=True)
class StateTensorInfo:
    input_name: str
    output_name: str
    kind: StateTensorKind


@dataclass(frozen=True, slots=True)
class ModelVariantId:
    model_id: ModelId
    variant_tag: str

    @property
    def full_variant_tag(self) -> str:
        return f"{self.model_id.model_name}-{self.variant_tag}"
