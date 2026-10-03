from pydantic import BaseModel

from integration.model_manager.dto.model_dto import ModelDto
from integration.model_manager.dto.model_variant_dto import ModelVariantDto
from shared.artifact.artifact_ref import ArtifactRef
from shared.model.model_partition import ModelPartitionId
from shared.model.model_variant import (
    ModelVariantId,
)


class RegisterModelRequest(BaseModel):
    model_dto: ModelDto


class StartModelVariantRegistrationRequest(BaseModel):
    model_variant_dto: ModelVariantDto


class StartModelVariantRegistrationResponse(BaseModel):
    artifact_ref: ArtifactRef


class FinalizeModelVariantRegistrationRequest(BaseModel):
    model_variant_id: ModelVariantId


class GenerateModelPartitionRequest(BaseModel):
    model_variant_id: ModelVariantId
    layers: list[str]


class GenerateModelPartitionResponse(BaseModel):
    model_partition_id: ModelPartitionId
    artifact_ref: ArtifactRef
