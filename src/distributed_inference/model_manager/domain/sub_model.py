from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict, field_validator

from distributed_inference.domain.identifiers import ServiceId
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


## This is the replica of a sub-model as defined by a single plan version
class SubModelReplicaId(BaseModel):
    model_config = ConfigDict(frozen=True)

    sub_model_id: SubModelId
    replica_id: int


## This is the deployment of a sub-model as defined by a single plan version
## The deployment is uniquely identified by the sub-model replica id and
## by the service it is deployed on
class SubModelDeploymentId(BaseModel):
    model_config = ConfigDict(frozen=True)

    sub_model_replica_id: SubModelReplicaId
    service_id: ServiceId
