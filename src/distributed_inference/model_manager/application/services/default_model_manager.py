from contextlib import AbstractAsyncContextManager
from pathlib import Path
from typing import Iterable, override

import aiofiles

from distributed_inference.artifact_materializer.application.ports.outbound.artifact_materializer import (
    ArtifactMaterializer,
)
from distributed_inference.artifact_processing.artifact_workspace import (
    ArtifactWorkspace,
    build_local_artifact_bundle_from_artifact_workspace,
)
from distributed_inference.artifact_store.application.ports.outbound.artifact_store import (
    ArtifactStore,
)
from distributed_inference.artifact_store.domain.artifact_key import (
    ModelVersionArtifactKey,
    SubModelArtifactKey,
)
from distributed_inference.artifact_store.domain.readable_artifact_bundle import (
    ReadableArtifactBundle,
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
from distributed_inference.model_manager.application.ports.inbound.model_manager import (
    ModelManager,
)
from distributed_inference.model_metadata_store.application.ports.outbound.model_metadata_store import (
    ModelMetadataStore,
)
from distributed_inference.model_profile.application.ports.inbound.model_profiler import (
    ModelProfiler,
)
from distributed_inference.model_splitter.application.ports.outbound.model_splitter import (
    ModelSplitter,
)


class DefaultModelManager(ModelManager):
    ## TODO: We should add rollback logick with rollback library

    def __init__(
        self,
        model_profiler: ModelProfiler,
        artifact_store: ArtifactStore,
        model_metadata_store: ModelMetadataStore,
        artifact_materializer: ArtifactMaterializer,
        model_splitter: ModelSplitter,
    ):
        self._model_profiler = model_profiler

        self._artifact_store = artifact_store
        self._artifact_materializer = artifact_materializer
        self._model_metadata_store = model_metadata_store

        self._model_splitter = model_splitter
        pass

    @override
    async def register_model(self, owner_id: UserId, model_name: str) -> ModelId:
        return await self._model_metadata_store.register_model(owner_id, model_name)

    @override
    async def put_model_version(
        self, model_id: ModelId, model_info: ModelInfo, bundle: ReadableArtifactBundle
    ) -> ModelVersionId:
        model_version_id = await self._model_metadata_store.register_model_version(
            model_id, model_info
        )
        artifact_key = ModelVersionArtifactKey(id=model_version_id)
        await self._artifact_store.put_artifact(artifact_key, bundle)

        async with self._artifact_materializer.materialize_artifact(
            artifact_key
        ) as materialized_artifact:
            model_graph = await self._model_profiler.profile_model(
                materialized_artifact, model_info
            )
            await self._model_metadata_store.register_model_version_graph(
                model_version_id, model_graph
            )
        return model_version_id

    @override
    async def generate_sub_model(
        self, model_version_id: ModelVersionId, layers: Iterable[LayerKey]
    ) -> SubModelId:

        SubModelId.check_valid_layers_format(layers)

        layers = tuple(layers)
        model_graph = await self._model_metadata_store.get_model_graph(model_version_id)
        if model_graph is None:
            raise ValueError(
                f"Model graph for model version {model_version_id} still not ready"
            )

        sub_model_id = await self._model_metadata_store.register_sub_model(
            model_version_id, layers
        )

        async with aiofiles.tempfile.TemporaryDirectory() as tmp_dir:
            split_artifact_paths = ArtifactWorkspace(
                root_path=Path(tmp_dir), entrypoint_path=None
            )

            artifact_key = ModelVersionArtifactKey(id=model_version_id)
            async with self._artifact_materializer.materialize_artifact(
                artifact_key
            ) as model_paths:
                await self._model_splitter.split_model(
                    model_graph, layers, model_paths, split_artifact_paths
                )

            sub_model_artifact_key = SubModelArtifactKey(id=sub_model_id)
            artifact_bundle = build_local_artifact_bundle_from_artifact_workspace(
                split_artifact_paths
            )
            await self._artifact_store.put_artifact(
                sub_model_artifact_key, artifact_bundle
            )

        return sub_model_id

    @override
    def get_sub_model(
        self, sub_model_id: SubModelId
    ) -> AbstractAsyncContextManager[ReadableArtifactBundle]:
        artifact_key = SubModelArtifactKey(id=sub_model_id)
        return self._artifact_store.open_artifact(artifact_key)

    @override
    async def get_model_graph(self, model_version_id: ModelVersionId) -> ModelGraph:
        model_graph = await self._model_metadata_store.get_model_graph(model_version_id)
        if model_graph is None:
            raise ValueError(
                f"Model graph for model version {model_version_id} still not ready"
            )
        return model_graph

    @override
    async def check_model_version_existence(
        self, model_version_id: ModelVersionId
    ) -> bool:
        exists_metadata = (
            await self._model_metadata_store.check_model_version_existence(
                model_version_id
            )
        )
        artifact_key = ModelVersionArtifactKey(id=model_version_id)
        exists_artifact = await self._artifact_store.check_artifact_existence(
            artifact_key
        )
        return exists_metadata and exists_artifact

    @override
    async def check_sub_model_existence(self, sub_model_id: SubModelId) -> bool:
        exists_metadata = await self._model_metadata_store.check_sub_model_existence(
            sub_model_id
        )
        artifact_key = SubModelArtifactKey(id=sub_model_id)
        exists_artifact = await self._artifact_store.check_artifact_existence(
            artifact_key
        )
        return exists_metadata and exists_artifact
