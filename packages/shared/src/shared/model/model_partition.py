from collections.abc import Iterable
from dataclasses import dataclass

from shared.model.keys import LayerKey
from shared.model.model_variant import ModelVariantId


@dataclass(frozen=True, slots=True)
class ModelPartitionId:
    model_variant_id: ModelVariantId
    layers: tuple[LayerKey, ...]

    def __post_init__(self) -> None:
        self.check_valid_layers_format(self.layers)
        layers = tuple(self.layers)
        if not layers:
            raise ValueError("SubModelId layers must not be empty")
        if len(layers) != len(set(layers)):
            raise ValueError("SubModelId layers must not contain duplicates")
        object.__setattr__(self, "layers", tuple(sorted(layers)))

    @classmethod
    def check_valid_layers_format(cls, layers: Iterable[LayerKey]) -> None:
        if isinstance(layers, (str, bytes)):
            raise ValueError("Layers must contain layer names")
