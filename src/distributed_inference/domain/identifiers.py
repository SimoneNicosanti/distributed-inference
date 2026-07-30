from typing import Iterable, Self, Tuple
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    field_validator,
    model_validator,
)

from distributed_inference.domain.model_graph_info import LayerKey

# We can have multiple users.
# - Each user can define multiple flows.
# - Each flow can specify the model (or task type) to be executed.
# - Each model has multiple versions.
# - Each model version can be divided in multiple components after the optimization
# - Then we have the artifacts as stored in the model store.


class ServerId(BaseModel):
    model_config = ConfigDict(frozen=True)

    server_id: UUID


class ServiceId(BaseModel):
    model_config = ConfigDict(frozen=True)

    server_id: ServerId
    service_id: UUID


class UserId(BaseModel):
    model_config = ConfigDict(frozen=True)

    user_id: UUID
    pass


class FlowId(BaseModel):
    model_config = ConfigDict(frozen=True)

    user_id: UserId
    flow_id: UUID
    pass


class RequestId(BaseModel):
    model_config = ConfigDict(frozen=True, validate_assignment=True)

    request_idx: UUID
    flow_id: FlowId
    sub_model_id: SubModelId

    @model_validator(mode="after")
    def validate_request_id(self) -> Self:

        if self.flow_id.user_id != self.sub_model_id.model_version_id.model_id.user_id:
            raise ValueError(
                f"Flow {self.flow_id} is not owned by user {self.sub_model_id.model_version_id.model_id.user_id}"
            )

        if self.request_idx is None:
            self.request_idx = UUID(int=0)

        return self

    pass


class ModelId(BaseModel):
    model_config = ConfigDict(frozen=True)

    user_id: UserId
    model_name: str

    @field_validator("model_name", mode="before")
    @classmethod
    def validate_model_name(cls, model_name: str) -> str:
        if model_name.find("/") != -1:
            raise ValueError("Model name cannot contain '/'")
        if model_name.find("\\") != -1:
            raise ValueError("Model name cannot contain '\\'")
        if model_name.find("..") != -1:
            raise ValueError("Model name cannot contain '..'")

        return model_name

    pass


class ModelVersionId(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_id: ModelId
    version_number: int
    pass


class SubModelDeploymentId(BaseModel):
    model_config = ConfigDict(frozen=True)

    sub_model_id: SubModelId
    service_id: ServiceId
    replica_id: int


class SubModelId(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_version_id: ModelVersionId
    layers: Tuple[LayerKey, ...]
    pass

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
