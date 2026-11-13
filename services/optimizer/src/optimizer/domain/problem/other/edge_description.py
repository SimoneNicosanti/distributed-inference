from dataclasses import dataclass, field

from shared.model.keys import LayerKey, TensorKey


@dataclass(frozen=True, slots=True)
class EdgeDescription:
    source: LayerKey
    target: LayerKey
    tensors: frozenset[TensorKey] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        object.__setattr__(self, "tensors", frozenset(self.tensors))
