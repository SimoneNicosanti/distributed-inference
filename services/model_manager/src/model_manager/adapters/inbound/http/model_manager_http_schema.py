from pydantic import BaseModel

from model_manager.domain.profiled_model_version import ProfiledModelVersion
from shared.model.model import Model, ModelId
from shared.model.model_version import (
    ModelVersionId,
)
from shared.model.sub_model import SubModel


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
