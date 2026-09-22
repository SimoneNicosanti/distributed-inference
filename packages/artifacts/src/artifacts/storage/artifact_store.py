"""Storage capability exposed by artifact backends."""

from abc import ABC, abstractmethod
from contextlib import AbstractAsyncContextManager

from artifacts.contracts.artifact_ref import ArtifactRef
from artifacts.workspace.artifact_bundle import ArtifactBundle


class ArtifactStore(ABC):
    @abstractmethod
    async def put_artifact(
        self,
        artifact_ref: ArtifactRef,
        bundle: ArtifactBundle,
    ) -> None: ...

    @abstractmethod
    def download_artifact(
        self,
        artifact_ref: ArtifactRef,
    ) -> AbstractAsyncContextManager[ArtifactBundle]:
        """Yield a local bundle whose paths remain valid for the context lifetime."""
        ...

    @abstractmethod
    async def check_artifact_existence(
        self,
        artifact_ref: ArtifactRef,
    ) -> bool: ...
