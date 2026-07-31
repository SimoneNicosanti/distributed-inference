from abc import ABC, abstractmethod
from contextlib import AbstractContextManager

from distributed_inference.domain.identifiers import (
    ModelVersionId,
)
from distributed_inference.model_artifact.domain.artifact_bundle import (
    ArtifactBundle,
)


class ModelVersionArtifactStore(ABC):
    @abstractmethod
    async def put_model_version(
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
    async def check_model_version_existence(
        self,
        model_version_id: ModelVersionId,
    ) -> bool: ...
