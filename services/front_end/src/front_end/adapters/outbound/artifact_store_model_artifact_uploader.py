from typing import override

from artifacts.contracts.artifact_bundle import ArtifactBundle
from artifacts.storage.artifact_store import ArtifactStore
from front_end.application.ports.outbound.model_loading_gateway import (
    ModelArtifactUploader,
)
from shared.artifact.artifact_ref import ArtifactRef


class ArtifactStoreModelArtifactUploader(ModelArtifactUploader):
    def __init__(self, artifact_store: ArtifactStore) -> None:
        self._artifact_store = artifact_store

    @override
    async def upload_model_artifact(
        self,
        artifact_ref: ArtifactRef,
        artifact_bundle: ArtifactBundle,
    ) -> None:
        await self._artifact_store.put_artifact(artifact_ref, artifact_bundle)
