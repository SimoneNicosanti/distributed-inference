from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from shared.model.keys import TensorKey
from worker.domain.tensor.tensor import Tensor


@dataclass(frozen=True, slots=True, eq=False)
class TensorBundle:
    bundle: Mapping[TensorKey, Tensor]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "bundle",
            MappingProxyType(dict(self.bundle)),
        )

    def get_tensor_by_name(self, name: str) -> Tensor | None:
        return self.bundle.get(name)

    def filter(self, names: list[str]) -> TensorBundle:
        return TensorBundle(
            bundle={
                name: tensor for name, tensor in self.bundle.items() if name in names
            }
        )

    def merge(self, other: TensorBundle) -> TensorBundle:
        if not isinstance(other, TensorBundle):
            raise TypeError("other must be a TensorBundle")
        return TensorBundle(bundle={**self.bundle, **other.bundle})

    def get_tensor_names(self) -> set[str]:
        return set(self.bundle)
