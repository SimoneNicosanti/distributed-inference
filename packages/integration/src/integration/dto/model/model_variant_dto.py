from typing import Annotated

from pydantic import BaseModel, Field, PositiveInt

from shared.model.model_variant import (
    ModelVariantFormat,
    NumericPrecision,
    PerformanceMetric,
    QuantizationType,
)


class DynamicShapeValuesDto(BaseModel):
    values: frozenset[Annotated[PositiveInt, Field(strict=True)]] = Field(min_length=1)


class InputInfoDto(BaseModel):
    shape: tuple[
        Annotated[str, Field(min_length=1)]
        | Annotated[PositiveInt, Field(strict=True)],
        ...,
    ]


class ModelVariantInfoDto(BaseModel):
    precision: NumericPrecision
    quantization: QuantizationType
    accuracies: list[PerformanceMetric]
    format: ModelVariantFormat

    inputs_info: dict[str, InputInfoDto]
    dynamic_dimension_values: dict[str, DynamicShapeValuesDto]
