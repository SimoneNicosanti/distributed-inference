from typing import Any, Tuple
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    field_validator,
)

from distributed_inference.domain.model_graph_info import LayerKey

# We can have multiple users.
# - Each user can define multiple flows.
# - Each flow can specify the model (or task type) to be executed.
# - Each model has multiple versions.
# - Each model version can be divided in multiple components after the optimization
# - Then we have the artifacts as stored in the model store.


class UserId(BaseModel):
    model_config = ConfigDict(frozen=True)

    user_id: UUID
    pass


class FlowId(BaseModel):
    model_config = ConfigDict(frozen=True)

    user_id: UserId
    flow_id: UUID
    pass


class ModelId(BaseModel):
    model_config = ConfigDict(frozen=True)

    user_id: UserId
    model_name: str
    pass


class ModelVersionId(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_id: ModelId
    version_number: int
    pass


class SubModelId(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_version_id: ModelVersionId
    layers: Tuple[LayerKey, ...]
    pass

    @field_validator("layers", mode="before")
    @classmethod
    def sort_layers(cls, value: Any) -> tuple[LayerKey, ...]:
        layers = tuple(value)

        if len(layers) != len(set(layers)):
            raise ValueError("SubModelId layers must not contain duplicates")

        return tuple(sorted(layers))
