import tempfile
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Iterable, override

from distributed_inference.application.model_artifact.contracts.materializer.model_version_materializer import (
    ModelVersionMaterializer,
)
from distributed_inference.application.model_artifact.contracts.store.model_version_artifact_store import (
    ModelVersionArtifactStore,
)
from distributed_inference.application.model_artifact.contracts.store.sub_model_artifact_store import (
    SubModelArtifactStore,
)
from distributed_inference.application.model_artifact.domain import (
    artifact_bundle_builder,
)
from distributed_inference.application.model_artifact.domain.artifact_bundle import (
    ArtifactBundle,
    ArtifactConcretePaths,
)
from distributed_inference.application.model_manager.contracts.model_manager import (
    ModelManager,
)
from distributed_inference.application.model_metadata_store.contracts.model_metadata_store import (
    ModelMetadataStore,
)
from distributed_inference.application.model_profile.contracts.model_profiler import (
    ModelProfiler,
)
from distributed_inference.application.model_splitter.contracts.model_splitter import (
    ModelSplitter,
)
from distributed_inference.domain.identifiers import (
    ModelId,
    ModelVersionId,
    SubModelId,
    UserId,
)
from distributed_inference.domain.model_graph_info import (
    LayerKey,
    ModelGraph,
    ModelInfo,
)


class DefaultModelManager(ModelManager):
    ## TODO: We should dd rollback logick with rollback library

    def __init__(
        self,
        model_profiler: ModelProfiler,
        model_version_artifact_store: ModelVersionArtifactStore,
        sub_model_artifact_store: SubModelArtifactStore,
        model_metadata_store: ModelMetadataStore,
        model_version_materializer: ModelVersionMaterializer,
        model_splitter: ModelSplitter,
    ):
        self._model_profiler = model_profiler

        self._model_version_artifact_store = model_version_artifact_store
        self._sub_model_artifact_store = sub_model_artifact_store
        self._model_metadata_store = model_metadata_store

        self._model_version_materializer = model_version_materializer

        self._model_splitter = model_splitter
        pass

    @override
    def register_model(self, owner_id: UserId, model_name: str) -> ModelId:
        return self._model_metadata_store.register_model(owner_id, model_name)

    @override
    def put_model_version(
        self, model_id: ModelId, model_info: ModelInfo, bundle: ArtifactBundle
    ) -> ModelVersionId:
        model_version_id = self._model_metadata_store.register_model_version(
            model_id, model_info
        )
        self._model_version_artifact_store.put_model_version(model_version_id, bundle)

        with self._model_version_materializer.materialize_model_version(
            model_version_id
        ) as artifact_concrete_paths:
            model_graph = self._model_profiler.profile_model(
                artifact_concrete_paths, model_info
            )
            self._model_metadata_store.register_model_version_graph(
                model_version_id, model_graph
            )
        return model_version_id

    @override
    def generate_sub_model(
        self, model_version_id: ModelVersionId, layers: Iterable[LayerKey]
    ) -> SubModelId:

        SubModelId.check_valid_layers_format(layers)

        layers = tuple(layers)
        model_graph = self._model_metadata_store.get_model_graph(model_version_id)
        if model_graph is None:
            raise ValueError(
                f"Model graph for model version {model_version_id} still not ready"
            )

        sub_model_id = self._model_metadata_store.register_sub_model(
            model_version_id, layers
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            split_artifact_paths = ArtifactConcretePaths(
                root_path=Path(tmp_dir), entrypoint_path=None
            )

            with self._model_version_materializer.materialize_model_version(
                model_version_id
            ) as model_paths:
                self._model_splitter.split_model(
                    model_graph, layers, model_paths, split_artifact_paths
                )

            with artifact_bundle_builder.build_artifact_bundle_from_bundle_paths(
                split_artifact_paths
            ) as artifact_bundle:
                self._sub_model_artifact_store.put_sub_model(
                    sub_model_id, artifact_bundle
                )

        return sub_model_id

    @override
    def get_sub_model(
        self, sub_model_id: SubModelId
    ) -> AbstractContextManager[ArtifactBundle]:
        return self._sub_model_artifact_store.get_sub_model(sub_model_id)

    @override
    def get_model_graph(self, model_version_id: ModelVersionId) -> ModelGraph:
        model_graph = self._model_metadata_store.get_model_graph(model_version_id)
        if model_graph is None:
            raise ValueError(
                f"Model graph for model version {model_version_id} still not ready"
            )
        return model_graph

    @override
    def check_model_version_existence(self, model_version_id: ModelVersionId) -> bool:
        exists_metadata = self._model_metadata_store.check_model_version_existence(
            model_version_id
        )
        exists_artifact = (
            self._model_version_artifact_store.check_model_version_existence(
                model_version_id
            )
        )
        return exists_metadata and exists_artifact

    @override
    def check_sub_model_existence(self, sub_model_id: SubModelId) -> bool:
        exists_metadata = self._model_metadata_store.check_sub_model_existence(
            sub_model_id
        )
        exists_artifact = self._sub_model_artifact_store.check_sub_model_existence(
            sub_model_id
        )
        return exists_metadata and exists_artifact
