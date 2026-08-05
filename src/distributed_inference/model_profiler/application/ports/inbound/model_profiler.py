from abc import ABC, abstractmethod

from distributed_inference.artifact_materializer.domain.materialized_artifact import (
    MaterializedArtifact,
)
from distributed_inference.model_manager.domain.model import ModelInfo
from distributed_inference.model_manager.domain.model_version import (
    ModelVersion,
    ProfiledModelVersion,
)


class ModelProfiler(ABC):
    @abstractmethod
    async def profile_model_version(
        self,
        artifact_concrete_paths: MaterializedArtifact,
        model_info: ModelInfo,
        model_version: ModelVersion,
    ) -> ProfiledModelVersion: ...
