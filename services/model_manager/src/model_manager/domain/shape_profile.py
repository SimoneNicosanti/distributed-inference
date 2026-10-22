import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from shared.model.keys import LayerKey, TensorKey


@dataclass(frozen=True, slots=True)
class InputShapePoint:
    name: str
    shape: tuple[int, ...]

    def __post_init__(self) -> None:
        if any(
            isinstance(dimension, bool) or dimension <= 0 for dimension in self.shape
        ):
            raise ValueError("Input shape dimensions must be positive integers")


## This is a dynamic shape configuration for which the profiling is done
## For each input, it tells a concrete shape it has
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
            raise ValueError("Tensor shape dimensions must be positive integers")


@dataclass(frozen=True, slots=True)
class ShapeProfile:
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
