from pydantic import BaseModel

from integration.dto.model.model_dto import ModelDto
from integration.dto.model.model_variant_dto import ModelVariantInfoDto
from shared.artifact.artifact_ref import ArtifactRef
from shared.model.model import ModelId
from shared.model.model_partition import ModelPartitionId
from shared.model.model_variant import (
    ModelVariantId,
)


class RegisterModelRequest(BaseModel):
    model_dto: ModelDto


class StartModelVariantRegistrationRequest(BaseModel):
    model_id: ModelId
    model_variant_info_dto: ModelVariantInfoDto


class StartModelVariantRegistrationResponse(BaseModel):
    model_variant_id: ModelVariantId
    artifact_ref: ArtifactRef


class FinalizeModelVariantRegistrationRequest(BaseModel):
    model_variant_id: ModelVariantId


class GenerateModelPartitionRequest(BaseModel):
    model_variant_id: ModelVariantId
    layers: list[str]


class GenerateModelPartitionResponse(BaseModel):
    model_partition_id: ModelPartitionId
    artifact_ref: ArtifactRef
