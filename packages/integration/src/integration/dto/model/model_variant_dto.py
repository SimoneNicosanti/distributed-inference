from pydantic import BaseModel, PositiveInt

from shared.model.model_variant import (
    ModelVariantFormat,
    NumericPrecision,
    PerformanceMetric,
    QuantizationType,
)


class DynamicShapeValuesDto(BaseModel):
    values: frozenset[PositiveInt]


class InputInfoDto(BaseModel):
    shape: tuple[str | PositiveInt, ...]


class ModelVariantInfoDto(BaseModel):
    precision: NumericPrecision
    quantization: QuantizationType
    accuracies: list[PerformanceMetric]
    format: ModelVariantFormat

    inputs_info: dict[str, InputInfoDto]
    dynamic_dimension_values: dict[str, DynamicShapeValuesDto]
