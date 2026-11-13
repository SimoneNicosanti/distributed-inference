from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import override

from shared.model.keys import TensorKey


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
