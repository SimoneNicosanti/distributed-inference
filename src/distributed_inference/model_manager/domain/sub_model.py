from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict, field_validator

from distributed_inference.model_manager.domain.model_version import ModelVersionId
from distributed_inference.model_manager.domain.model_version_graph import LayerKey


class SubModelId(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_version_id: ModelVersionId
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


class SubModel(BaseModel):
    model_config = ConfigDict(frozen=True)

    sub_model_id: SubModelId
