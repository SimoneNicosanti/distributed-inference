from abc import ABC, abstractmethod
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from pathlib import PurePosixPath

from distributed_inference.artifact_store.domain.artifact_manifest import (
    ArtifactManifest,
)


## This is used for the async read from the source
class AsyncByteReader(ABC):
    @abstractmethod
    async def read(self, size: int = -1) -> bytes: ...


## This is the artifact readable bundle; each backend technology decides how to handle the streaming
@dataclass(frozen=True)
class ReadableArtifactBundle(ABC):
    manifest: ArtifactManifest

    def get_manifest(self) -> ArtifactManifest:
        return self.manifest

    @abstractmethod
    def open_file(
        self, path: PurePosixPath
    ) -> AbstractAsyncContextManager[AsyncByteReader]: ...
