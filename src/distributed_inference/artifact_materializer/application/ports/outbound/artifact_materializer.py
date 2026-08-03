from abc import ABC, abstractmethod
from contextlib import AbstractAsyncContextManager

from distributed_inference.artifact_materializer.domain.materialized_artifact import (
    MaterializedArtifact,
)
from distributed_inference.artifact_store.domain.artifact_key import ArtifactKey


class ArtifactMaterializer(ABC):
    @abstractmethod
    def materialize_artifact(
        self,
        artifact_key: ArtifactKey,
    ) -> AbstractAsyncContextManager[MaterializedArtifact]: ...
