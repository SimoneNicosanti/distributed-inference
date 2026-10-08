from pydantic import BaseModel, PositiveInt

from shared.model.model_variant import (
    AccuracyMetric,
    ModelVariantFormat,
    ModelVariantId,
    NumericPrecision,
    QuantizationType,
)


class DynamicShapeValuesDto(BaseModel):
    values: frozenset[PositiveInt]


class InputInfoDto(BaseModel):
    shape: tuple[str | PositiveInt, ...]


class ModelVariantDto(BaseModel):
    id: ModelVariantId

    precision: NumericPrecision
    quantization: QuantizationType
    accuracies: list[AccuracyMetric]
    format: ModelVariantFormat

    inputs_info: dict[str, InputInfoDto]
    dynamic_dimension_values: dict[str, DynamicShapeValuesDto]
