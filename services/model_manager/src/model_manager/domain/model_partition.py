from pydantic import BaseModel, ConfigDict

from shared.artifact.artifact_ref import ArtifactRef
from shared.model.model_partition import ModelPartitionId


class ModelPartition(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_partition_id: ModelPartitionId
    artifact_ref: ArtifactRef
