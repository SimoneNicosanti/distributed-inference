from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True, slots=True)
class ShapePoint:
    input_shape_points: Mapping[str, tuple[int, ...]]

    def __post_init__(self) -> None:
        if any(
            isinstance(dimension, bool) or dimension <= 0
            for shape in self.input_shape_points.values()
            for dimension in shape
        ):
            raise ValueError("Input shape dimensions must be positive integers")
        object.__setattr__(
            self,
            "input_shape_points",
            MappingProxyType(dict(self.input_shape_points)),
        )
