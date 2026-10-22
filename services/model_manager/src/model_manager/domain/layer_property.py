from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from shared.model.keys import TensorKey


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
