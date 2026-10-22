from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from shared.model.model import ModelId
from shared.model.model_variant import (
    ModelVariantFormat,
    ModelVariantId,
    NumericPrecision,
    PerformanceMetric,
    QuantizationType,
)


@dataclass(frozen=True, slots=True)
class DynamicShapeValues:
    values: frozenset[int]

    def __post_init__(self) -> None:
        if any(isinstance(value, bool) or value <= 0 for value in self.values):
            raise ValueError("Dynamic shape values must be positive integers")


@dataclass(frozen=True, slots=True)
class InputInfo:
    shape: tuple[str | int, ...]  ## int --> Static ; str --> Dynamic

    def __post_init__(self) -> None:
        if any(
            isinstance(dimension, bool)
            or (isinstance(dimension, int) and dimension <= 0)
            for dimension in self.shape
        ):
            raise ValueError("Static input dimensions must be positive integers")


@dataclass(frozen=True, slots=True)
class ModelVariantInfo:
    precision: NumericPrecision
    quantization: QuantizationType
    accuracies: tuple[PerformanceMetric, ...]
    format: ModelVariantFormat

    inputs_info: Mapping[str, InputInfo]
    dynamic_dimension_values: Mapping[str, DynamicShapeValues]

    def __post_init__(self) -> None:
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

        object.__setattr__(
            self, "inputs_info", MappingProxyType(dict(self.inputs_info))
        )
        object.__setattr__(
            self,
            "dynamic_dimension_values",
            MappingProxyType(dict(self.dynamic_dimension_values)),
        )


@dataclass(frozen=True, slots=True)
class ModelVariant:
    model_variant_id: ModelVariantId
    model_variant_info: ModelVariantInfo

    @property
    def model_id(self) -> ModelId:
        return self.model_variant_id.model_id
