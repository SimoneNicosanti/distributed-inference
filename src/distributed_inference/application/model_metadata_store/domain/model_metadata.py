from pydantic import BaseModel, ConfigDict

from distributed_inference.domain.identifiers import (
    ModelId,
    ModelVersionId,
    SubModelId,
    UserId,
)
from distributed_inference.domain.model_graph_info import ModelGraph, ModelInfo


class ModelMetadata(BaseModel):
    model_config = ConfigDict(frozen=True)

    owner_id: UserId
    model_id: ModelId
    name: str


class ModelVersionMetadata(BaseModel):
    model_config = ConfigDict(frozen=False)

    model_id: ModelId
    model_version_id: ModelVersionId

    version_number: int

    model_info: ModelInfo
    model_graph: ModelGraph | None = None


class SubModelMetadata(BaseModel):
    model_config = ConfigDict(frozen=True)

    sub_model_id: SubModelId
