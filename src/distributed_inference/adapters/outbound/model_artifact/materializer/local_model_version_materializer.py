from contextlib import AbstractContextManager
from typing import override

from distributed_inference.adapters.outbound.model_artifact.store.local.local_model_version_artifact_store import (
    LocalModelVersionArtifactStore,
)
from distributed_inference.application.model_artifact.contracts.materializer.model_version_materializer import (
    ModelVersionMaterializer,
)
from distributed_inference.application.model_artifact.domain.artifact_bundle import (
    MaterializedArtifact,
)
from distributed_inference.domain.identifiers import ModelVersionId


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
    ) -> AbstractContextManager[MaterializedArtifact]:
        return self._local_model_version_artifact_store.get_model_version_bundle_path(
            model_version_id
        )
