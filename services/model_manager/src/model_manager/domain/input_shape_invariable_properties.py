from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from shared.model.keys import LayerKey, TensorKey


@dataclass(frozen=True, slots=True)
class InputShapeInvariableTensorProperties:
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
class InputShapeInvariableLayerProperties:
    weight_properties: Mapping[TensorKey, InputShapeInvariableTensorProperties]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "weight_properties",
            MappingProxyType(dict(self.weight_properties)),
        )


@dataclass(frozen=True, slots=True)
class InputShapeInvariableProperties:
    layer_properties: Mapping[LayerKey, InputShapeInvariableLayerProperties]
    tensor_properties: Mapping[TensorKey, InputShapeInvariableTensorProperties]

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
