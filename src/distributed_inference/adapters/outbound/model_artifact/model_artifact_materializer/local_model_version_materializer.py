from distributed_inference.application.model_artifact.contracts.materializer.model_version_materializer import (
    ModelVersionMaterializer,
)

from distributed_inference.adapters.outbound.model_artifact.model_artifact_store.local_model_version_store import (
    LocalModelVersionArtifactStore,
)

from distributed_inference.domain.identifiers import ModelVersionId

from typing import override

from contextlib import contextmanager

from pathlib import Path

from collections.abc import Generator


class LocalModelVersionMaterializer(ModelVersionMaterializer):
    def __init__(
        self,
        model_artifact_store: LocalModelVersionArtifactStore,
    ) -> None:
        self._local_model_version_artifact_store = model_artifact_store
        pass

    @override
    @contextmanager
    def materialize_model(
        self,
        model_version_id: ModelVersionId,
    ) -> Generator[Path]:
        with self._local_model_version_artifact_store.get_model_version_path(
            model_version_id
        ) as model_path:
            yield model_path
