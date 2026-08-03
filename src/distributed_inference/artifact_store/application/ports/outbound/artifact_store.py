from abc import ABC, abstractmethod
from contextlib import AbstractAsyncContextManager

from distributed_inference.artifact_store.domain.artifact_key import ArtifactKey
from distributed_inference.artifact_store.domain.readable_artifact_bundle import (
    ReadableArtifactBundle,
)


class ArtifactStore(ABC):
    @abstractmethod
    async def put_artifact(
        self,
        artifact_key: ArtifactKey,
        readable_bundle: ReadableArtifactBundle,
    ) -> None: ...

    @abstractmethod
    def open_artifact(
        self,
        artifact_key: ArtifactKey,
    ) -> AbstractAsyncContextManager[ReadableArtifactBundle]: ...

    @abstractmethod
    async def check_artifact_existence(
        self,
        artifact_key: ArtifactKey,
    ) -> bool: ...
