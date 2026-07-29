from abc import ABC, abstractmethod
from contextlib import AbstractContextManager

from distributed_inference.domain.identifiers import (
    ModelVersionId,
)
from distributed_inference.model_artifact.domain.artifact_bundle import (
    ArtifactConcretePaths,
)


class ModelVersionMaterializer(ABC):
    @abstractmethod
    def materialize_model_version(
        self,
        model_version_id: ModelVersionId,
    ) -> AbstractContextManager[ArtifactConcretePaths]: ...
