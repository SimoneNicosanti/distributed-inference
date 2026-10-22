from abc import ABC, abstractmethod

from shared.artifact.artifact_ref import ArtifactRef
from shared.model.model_partition import ModelPartitionId
from shared.model.model_variant import ModelVariantId


class ArtifactRefFactory(ABC):
    @abstractmethod
    async def build_artifact_ref(
        self, value: ModelVariantId | ModelPartitionId
    ) -> ArtifactRef: ...
