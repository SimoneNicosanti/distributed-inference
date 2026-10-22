import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from controller.domain.profile.model_variant_topology import ModelVariantTopology
from shared.model.keys import LayerKey, TensorKey
from shared.model.model import ModelTask, ModelVisibility
from shared.model.model_variant import ModelVariantId, PerformanceMetric


@dataclass(frozen=True, slots=True)
class ModelVariantInfo:
    visibility: ModelVisibility
    task: ModelTask

    accuracies: tuple[PerformanceMetric, ...]


@dataclass(frozen=True, slots=True)
class WeightProperty:
    data_type: str
    shape: tuple[int, ...]
    data_type_size_in_bytes: int

    def __post_init__(self) -> None:
        if self.data_type_size_in_bytes <= 0:
            raise ValueError("Data type size must be positive")


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
        if self.data_type_size_in_bytes <= 0:
            raise ValueError("Data type size must be positive")


@dataclass(frozen=True, slots=True)
class InputShapePoint:
    name: str
    shape: tuple[int, ...]

    def __post_init__(self) -> None:
        if any(
            isinstance(dimension, bool) or dimension <= 0 for dimension in self.shape
        ):
            raise ValueError("Input shape dimensions must be positive")


@dataclass(frozen=True, slots=True)
class ShapePoint:
    input_shape_points: tuple[InputShapePoint, ...]


@dataclass(frozen=True, slots=True)
class LayerShapeProperty:
    flops: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.flops) or self.flops < 0:
            raise ValueError("FLOPs must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class TensorShapeProperty:
    shape: tuple[int, ...]

    def __post_init__(self) -> None:
        if any(
            isinstance(dimension, bool) or dimension <= 0 for dimension in self.shape
        ):
            raise ValueError("Tensor shape dimensions must be positive")


@dataclass(frozen=True, slots=True)
class ShapeProfile:
    shape_point: ShapePoint
    layer_properties: Mapping[LayerKey, LayerShapeProperty]
    tensor_properties: Mapping[TensorKey, TensorShapeProperty]

    def __post_init__(self) -> None:
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
