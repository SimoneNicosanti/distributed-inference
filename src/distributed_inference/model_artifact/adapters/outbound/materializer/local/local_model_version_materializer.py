from contextlib import AbstractContextManager
from typing import override

from distributed_inference.domain.identifiers import ModelVersionId
from distributed_inference.model_artifact.adapters.outbound.store.local.local_model_version_artifact_store import (
    LocalModelVersionArtifactStore,
)
from distributed_inference.model_artifact.application.ports.outbound.materializer.model_version_materializer import (
    ModelVersionMaterializer,
)
from distributed_inference.model_artifact.domain.artifact_bundle import (
    ArtifactConcretePaths,
)


class LocalModelVersionMaterializer(ModelVersionMaterializer):
    def __init__(
        self,
        model_artifact_store: LocalModelVersionArtifactStore,
    ) -> None:
        self._local_model_version_artifact_store = model_artifact_store
        pass

    @override
    def materialize_model_version(
        self,
        model_version_id: ModelVersionId,
    ) -> AbstractContextManager[ArtifactConcretePaths]:
        return self._local_model_version_artifact_store.get_model_version_bundle_path(
            model_version_id
        )
