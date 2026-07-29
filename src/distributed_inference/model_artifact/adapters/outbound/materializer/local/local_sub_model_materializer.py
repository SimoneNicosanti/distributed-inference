from contextlib import AbstractContextManager
from typing import override

from distributed_inference.domain.identifiers import SubModelId
from distributed_inference.model_artifact.adapters.outbound.store.local.local_sub_model_artifact_store import (
    LocalSubModelArtifactStore,
)
from distributed_inference.model_artifact.application.ports.outbound.materializer.sub_model_materializer import (
    SubModelMaterializer,
)
from distributed_inference.model_artifact.domain.artifact_bundle import (
    ArtifactConcretePaths,
)


class LocalSubModelMaterializer(SubModelMaterializer):
    def __init__(
        self,
        sub_model_artifact_store: LocalSubModelArtifactStore,
    ) -> None:
        self._local_sub_model_artifact_store = sub_model_artifact_store
        pass

    @override
    def materialize_sub_model(
        self,
        sub_model_id: SubModelId,
    ) -> AbstractContextManager[ArtifactConcretePaths]:
        return self._local_sub_model_artifact_store.get_sub_model_path(sub_model_id)
