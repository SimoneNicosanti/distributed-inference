import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import override

from controller.domain.profile.model_variant_topology import ModelVariantTopology
from shared.model.keys import LayerKey, TensorKey
from shared.model.model import ModelTask
from shared.model.model_variant import ModelVariantId, PerformanceMetric


@dataclass(frozen=True, slots=True)
class ModelVariantInfo:
    task: ModelTask

    accuracies: tuple[PerformanceMetric, ...]


@dataclass(frozen=True, slots=True)
class WeightProperty:
    data_type: str
    shape: tuple[int, ...]
    data_type_size_in_bytes: int

    def __post_init__(self) -> None:
        if (
            type(self.data_type_size_in_bytes) is not int
            or self.data_type_size_in_bytes <= 0
        ):
            raise ValueError("Data type size must be a positive integer byte count")
        if any(
            type(dimension) is not int or dimension <= 0 for dimension in self.shape
        ):
            raise ValueError("Weight shape dimensions must be positive integers")


@dataclass(frozen=True, slots=True)
class LayerProperty:
    weight_properties: Mapping[TensorKey, WeightProperty]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "weight_properties",
            MappingProxyType(dict(self.weight_properties)),
        )


@dataclass(frozen=True, slots=True)
class TensorProperty:
    data_type: str
    data_type_size_in_bytes: int
    shape_expression: tuple[int | str, ...]

    def __post_init__(self) -> None:
        if (
            type(self.data_type_size_in_bytes) is not int
            or self.data_type_size_in_bytes <= 0
        ):
            raise ValueError("Data type size must be a positive integer byte count")
        if any(
            not dimension
            if isinstance(dimension, str)
            else (type(dimension) is not int or dimension <= 0)
            for dimension in self.shape_expression
        ):
            raise ValueError(
                "Tensor dimensions must be positive integers or non-empty symbols"
            )


@dataclass(frozen=True, slots=True)
class ModelInputConcreteShapes:
    input_shapes: Mapping[TensorKey, tuple[int, ...]]

    def __post_init__(self) -> None:
        if any(
            type(dimension) is not int or dimension <= 0
            for shape in self.input_shapes.values()
            for dimension in shape
        ):
            raise ValueError("Input shape dimensions must be positive")
        object.__setattr__(
            self, "input_shapes", MappingProxyType(dict(self.input_shapes))
        )

    @override
    def __hash__(self) -> int:
        return hash(frozenset(self.input_shapes.items()))


@dataclass(frozen=True, slots=True)
class LayerShapeProperty:
    flops: float

    def __post_init__(self) -> None:
        if (
            type(self.flops) not in (int, float)
            or not math.isfinite(self.flops)
            or self.flops < 0
        ):
            raise ValueError("FLOPs must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class TensorShapeProperty:
    shape: tuple[int, ...]

    def __post_init__(self) -> None:
        if any(
            type(dimension) is not int or dimension <= 0 for dimension in self.shape
        ):
            raise ValueError("Tensor shape dimensions must be positive")


@dataclass(frozen=True, slots=True)
class ShapeProfile:
    input_shapes: ModelInputConcreteShapes
    layer_properties: Mapping[LayerKey, LayerShapeProperty]
    tensor_properties: Mapping[TensorKey, TensorShapeProperty]
    max_cut_size_mib: float | None = None

    def __post_init__(self) -> None:
        if self.max_cut_size_mib is not None and (
            not math.isfinite(self.max_cut_size_mib) or self.max_cut_size_mib <= 0
        ):
            raise ValueError("max_cut_size_mib must be finite and positive")
        object.__setattr__(
            self,
            "layer_properties",
            MappingProxyType(dict(self.layer_properties)),
        )
        object.__setattr__(
            self,
            "tensor_properties",
            MappingProxyType(dict(self.tensor_properties)),
        )


@dataclass(frozen=True, slots=True)
class ModelVariantStaticProfile:
    model_variant_id: ModelVariantId

    info: ModelVariantInfo
    topology: ModelVariantTopology
    invariant_layer_properties: Mapping[LayerKey, LayerProperty]
    invariant_tensor_properties: Mapping[TensorKey, TensorProperty]
    shape_profiles: tuple[ShapeProfile, ...]
    optimization_contractions: tuple[tuple[LayerKey, ...], ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "invariant_layer_properties",
            MappingProxyType(dict(self.invariant_layer_properties)),
        )
        object.__setattr__(
            self,
            "invariant_tensor_properties",
            MappingProxyType(dict(self.invariant_tensor_properties)),
        )
