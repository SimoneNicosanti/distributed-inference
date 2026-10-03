from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict, field_validator

from shared.model.keys import LayerKey
from shared.model.model_variant import ModelVariantId


class ModelPartitionId(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_variant_id: ModelVariantId
    layers: tuple[LayerKey, ...]

    @field_validator("layers")
    @classmethod
    def validate_layers(
        cls,
        layers: tuple[LayerKey, ...],
    ) -> tuple[LayerKey, ...]:
        if not layers:
            raise ValueError("SubModelId layers must not be empty")

        if len(layers) != len(set(layers)):
            raise ValueError("SubModelId layers must not contain duplicates")

        return tuple(sorted(layers))

    @classmethod
    def check_valid_layers_format(cls, layers: Iterable[LayerKey]) -> None:
        if isinstance(layers, (str, bytes)):
            raise ValueError("Layers must contain layer names")
