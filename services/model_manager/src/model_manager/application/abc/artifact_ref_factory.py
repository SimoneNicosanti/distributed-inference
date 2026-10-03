from abc import ABC, abstractmethod

from shared.artifact.artifact_ref import ArtifactRef


class ArtifactRefFactory(ABC):
    @abstractmethod
    async def build_artifact_ref(self, value: str) -> ArtifactRef: ...
