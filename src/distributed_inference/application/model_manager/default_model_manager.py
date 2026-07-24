import tempfile
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO, Iterable, override

from distributed_inference.application.model_artifact.contracts.materializer.model_version_materializer import (
    ModelVersionMaterializer,
)
from distributed_inference.application.model_artifact.contracts.store.model_version_artifact_store import (
    ModelVersionArtifactStore,
)
from distributed_inference.application.model_artifact.contracts.store.sub_model_artifact_store import (
    SubModelArtifactStore,
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
    pass

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
        pass

    @override
    def upload_model_version(
        self, model_id: ModelId, model_info: ModelInfo, binary_io: BinaryIO
    ) -> ModelVersionId:
        model_version_id = self._model_metadata_store.register_model_version(
            model_id, model_info
        )
        self._model_version_artifact_store.put_model_version(
            model_version_id, binary_io
        )

        with self._model_version_materializer.materialize_model_version(
            model_version_id
        ) as model_path:
            model_graph = self._model_profiler.profile_model(model_path, model_info)
            self._model_metadata_store.register_model_version_graph(
                model_version_id, model_graph
            )
        return model_version_id

    @override
    def generate_sub_model(
        self, version_id: ModelVersionId, layers: Iterable[LayerKey]
    ) -> SubModelId:

        model_graph = self._model_metadata_store.get_model_graph(version_id)
        sub_model_id = self._model_metadata_store.register_sub_model(version_id, layers)

        with tempfile.TemporaryDirectory() as tmp_dir:
            split_model_path = Path(tmp_dir) / "split_model.onnx"
            with self._model_version_materializer.materialize_model_version(
                version_id
            ) as model_path:
                self._model_splitter.split_model(
                    model_graph, layers, model_path, split_model_path
                )

            with split_model_path.open("rb") as binary_io:
                self._sub_model_artifact_store.put_sub_model(sub_model_id, binary_io)

        return sub_model_id

    @override
    @contextmanager
    def download_sub_model(self, submodel_id: SubModelId) -> Generator[BinaryIO]:
        with self._sub_model_artifact_store.get_sub_model(submodel_id) as binary_io:
            yield binary_io

    @override
    def get_model_graph(self, model_version_id: ModelVersionId) -> ModelGraph:
        return self._model_metadata_store.get_model_graph(model_version_id)

    @override
    def check_model_version_existence(self, model_version_id: ModelVersionId) -> bool:
        return True

    @override
    def check_sub_model_existence(self, submodel_id: SubModelId) -> bool:
        return True
