from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from typing_extensions import override

from distributed_inference.artifact_materializer.application.ports.outbound.artifact_materializer import (
    ArtifactMaterializer,
)
from distributed_inference.artifact_materializer.domain.materialized_artifact import (
    MaterializedArtifact,
)
from distributed_inference.artifact_store.adapters.outbound.local.local_artifact_store import (
    LocalArtifactStore,
)
from distributed_inference.artifact_store.domain.artifact_key import ArtifactKey


class LocalArtifactMaterializer(ArtifactMaterializer):
    def __init__(self, artifact_store: LocalArtifactStore):
        self._artifact_store = artifact_store

    @override
    @asynccontextmanager
    async def materialize_artifact(
        self,
        artifact_key: ArtifactKey,
    ) -> AsyncGenerator[MaterializedArtifact]:

        async with self._artifact_store.get_artifact_manifest_root_path_entry_path(
            artifact_key
        ) as artifact_tuple:
            manifest, artifact_root_path, entrypoint_path = artifact_tuple
            yield MaterializedArtifact(
                manifest=manifest,
                root_path=artifact_root_path,
                entrypoint_path=entrypoint_path,
            )
