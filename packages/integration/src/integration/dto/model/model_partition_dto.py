from pydantic import BaseModel

from shared.artifact.artifact_ref import ArtifactRef
from shared.model.model_partition import ModelPartitionId


class ModelPartitionDto(BaseModel):
    model_partition_id: ModelPartitionId
    artifact_ref: ArtifactRef
