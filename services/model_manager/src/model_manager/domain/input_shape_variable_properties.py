import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import override

from shared.model.keys import LayerKey, TensorKey


## This is a dynamic shape configuration for which the profiling is done
## For each input, it tells a concrete shape it has
@dataclass(frozen=True, slots=True)
class ModelInputConcreteShapes:
    input_shapes: Mapping[TensorKey, tuple[int, ...]]

    def __post_init__(self) -> None:
        for input_shape in self.input_shapes.values():
            for dimension in input_shape:
                if type(dimension) is not int or dimension <= 0:
                    raise ValueError(
                        "Concrete shape dimensions must be positive integers"
                    )

        object.__setattr__(
            self,
            "input_shapes",
            MappingProxyType(dict(self.input_shapes)),
        )

    @override
    def __hash__(self) -> int:
        return hash(frozenset(self.input_shapes.items()))


@dataclass(frozen=True, slots=True)
class InputShapeVariableLayerProperties:
    flops: float

    def __post_init__(self) -> None:
        if (
            type(self.flops) not in (int, float)
            or not math.isfinite(self.flops)
            or self.flops < 0
        ):
            raise ValueError("FLOPs must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class InputShapeVariableTensorProperties:
    shape: tuple[int, ...]

    def __post_init__(self) -> None:
        if any(
            type(dimension) is not int or dimension <= 0 for dimension in self.shape
        ):
            raise ValueError("Tensor shape dimensions must be positive integers")


@dataclass(frozen=True, slots=True)
class InputShapeVariableModelProperties:
    max_cut_size_mib: float

    def __post_init__(self) -> None:
        if (
            type(self.max_cut_size_mib) not in (int, float)
            or not math.isfinite(self.max_cut_size_mib)
            or self.max_cut_size_mib <= 0
        ):
            raise ValueError("Max cut size must be finite and positive (MiB)")


@dataclass(frozen=True, slots=True)
class InputShapeVariableProperties:
    layer_properties: Mapping[LayerKey, InputShapeVariableLayerProperties]
    tensor_properties: Mapping[TensorKey, InputShapeVariableTensorProperties]
    model_properties: InputShapeVariableModelProperties

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
