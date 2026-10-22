from dataclasses import dataclass

from shared.artifact.artifact_ref import ArtifactRef
from shared.model.model_partition import ModelPartitionId


@dataclass(frozen=True, slots=True)
class ModelPartition:
    model_partition_id: ModelPartitionId
    artifact_ref: ArtifactRef
