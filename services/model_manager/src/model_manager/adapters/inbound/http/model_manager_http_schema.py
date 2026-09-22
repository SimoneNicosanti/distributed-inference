from pydantic import BaseModel

from integration.model.model import Model, ModelId
from integration.model.model_version import (
    ModelVersionId,
)
from integration.model.sub_model import SubModel
from model_manager.domain.profiled_model_version import ProfiledModelVersion


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
