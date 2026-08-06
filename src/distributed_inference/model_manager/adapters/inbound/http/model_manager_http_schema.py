from pydantic import BaseModel

from distributed_inference.model_manager.domain.model import Model, ModelId
from distributed_inference.model_manager.domain.model_version import (
    ModelVersionId,
    ProfiledModelVersion,
)
from distributed_inference.model_manager.domain.sub_model import SubModel


class RegisterModelRequest(BaseModel):
    model: Model


class RegisterModelResponse(BaseModel):
    model_id: ModelId


class UploadModelVersionResponse(BaseModel):
    model_version_id: ModelVersionId


class GenerateSubModelRequest(BaseModel):
    model_version_id: ModelVersionId
    layers: list[str]


class GenerateSubModelResponse(BaseModel):
    sub_model: SubModel


class GetProfiledModelVersionRequest(BaseModel):
    model_version_id: ModelVersionId


class GetProfiledModelVersionResponse(BaseModel):
    profiled_model_version: ProfiledModelVersion
