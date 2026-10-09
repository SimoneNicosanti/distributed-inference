from abc import ABC, abstractmethod

from shared.artifact.artifact_ref import ArtifactRef
from shared.model.model_partition import ModelPartitionId


class ModelPartitionMetadataReader(ABC):
    @abstractmethod
    async def get_artifact_ref_by_partition_id(
        self, partition_id: ModelPartitionId
    ) -> ArtifactRef: ...
