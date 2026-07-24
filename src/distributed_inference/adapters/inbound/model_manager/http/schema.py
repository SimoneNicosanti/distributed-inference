from pydantic import BaseModel

from distributed_inference.domain.identifiers import (
    ModelId,
    ModelVersionId,
    SubModelId,
    UserId,
)
from distributed_inference.domain.model_graph_info import ModelGraph


class RegisterModelRequest(BaseModel):
    owner_id: UserId
    model_name: str


class RegisterModelResponse(BaseModel):
    model_id: ModelId


class UploadModelVersionResponse(BaseModel):
    model_version_id: ModelVersionId


class GenerateSubModelRequest(BaseModel):
    model_version_id: ModelVersionId
    layers: list


class GenerateSubModelResponse(BaseModel):
    sub_model_id: SubModelId


class DownloadSubModelRequest(BaseModel):
    sub_model_id: SubModelId


class GetModelGraphRequest(BaseModel):
    model_version_id: ModelVersionId


class GetModelGraphResponse(BaseModel):
    model_graph: ModelGraph
