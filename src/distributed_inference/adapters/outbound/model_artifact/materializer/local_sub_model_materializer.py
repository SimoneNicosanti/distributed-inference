from distributed_inference.application.model_artifact.contracts.materializer.sub_model_materializer import (
    SubModelMaterializer,
)

from distributed_inference.adapters.outbound.model_artifact.store.local_sub_model_artifact_store import (
    LocalSubModelArtifactStore,
)

from distributed_inference.domain.identifiers import SubModelId

from typing import override

from contextlib import contextmanager

from pathlib import Path

from collections.abc import Generator


class LocalSubModelMaterializer(SubModelMaterializer):
    def __init__(
        self,
        sub_model_artifact_store: LocalSubModelArtifactStore,
    ) -> None:
        self._local_sub_model_artifact_store = sub_model_artifact_store
        pass

    @override
    @contextmanager
    def materialize_sub_model(
        self,
        sub_model_id: SubModelId,
    ) -> Generator[Path]:
        with self._local_sub_model_artifact_store.get_sub_model_path(
            sub_model_id
        ) as sub_model_path:
            yield sub_model_path
