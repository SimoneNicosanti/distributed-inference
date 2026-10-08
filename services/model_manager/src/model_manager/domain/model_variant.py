from typing import Self

from pydantic import BaseModel, ConfigDict, PositiveInt, model_validator

from shared.model.model import ModelId
from shared.model.model_variant import (
    AccuracyMetric,
    ModelVariantFormat,
    ModelVariantId,
    NumericPrecision,
    QuantizationType,
)


class DynamicShapeValues(BaseModel):
    model_config = ConfigDict(frozen=True)

    values: frozenset[PositiveInt]


class InputInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    shape: tuple[str | PositiveInt, ...]  ## int --> Static ; str --> Dynamic


class ModelVariantInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    precision: NumericPrecision
    quantization: QuantizationType
    accuracies: list[AccuracyMetric]
    format: ModelVariantFormat

    inputs_info: dict[str, InputInfo]
    dynamic_dimension_values: dict[str, DynamicShapeValues]

    @model_validator(mode="after")
    def validate_input_shapes(self) -> Self:
        for input_name, input_info in self.inputs_info.items():
            input_shape = input_info.shape
            for shape in input_shape:
                if isinstance(shape, int):
                    continue
                else:
                    if shape not in self.dynamic_dimension_values:
                        raise ValueError(
                            f"Unknown dynamic dimension {shape!r} in input {input_name!r}"
                        )

        return self


class ModelVariant(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_variant_id: ModelVariantId
    model_variant_info: ModelVariantInfo

    @property
    def model_id(self) -> ModelId:
        return self.model_variant_id.model_id
