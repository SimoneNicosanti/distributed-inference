from abc import ABC, abstractmethod
from contextlib import AbstractContextManager

from distributed_inference.application.model_artifact.domain.artifact_bundle import (
    ArtifactBundle,
)
from distributed_inference.domain.identifiers import (
    ModelVersionId,
)


class ModelVersionArtifactStore(ABC):
    @abstractmethod
    def put_model_version(
        self,
        model_version_id: ModelVersionId,
        bundle: ArtifactBundle,
    ) -> None: ...

    @abstractmethod
    def get_model_version(
        self,
        model_version_id: ModelVersionId,
    ) -> AbstractContextManager[ArtifactBundle]: ...

    @abstractmethod
    def check_model_version_existance(
        self,
        artifact_id: ModelVersionId,
    ) -> bool: ...
