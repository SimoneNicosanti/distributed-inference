from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True, slots=True)
class ModelInputConcreteShapes:
    input_shapes: Mapping[str, tuple[int, ...]]

    def __post_init__(self) -> None:
        if any(
            type(dimension) is not int or dimension <= 0
            for shape in self.input_shapes.values()
            for dimension in shape
        ):
            raise ValueError("Input shape dimensions must be positive integers")
        object.__setattr__(
            self,
            "input_shapes",
            MappingProxyType(dict(self.input_shapes)),
        )
