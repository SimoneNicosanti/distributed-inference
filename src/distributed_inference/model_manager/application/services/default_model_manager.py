from collections.abc import Iterable
from pathlib import Path
from typing import override

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
from distributed_inference.model_manager.application.ports.inbound.model_manager import (
    ModelManager,
)
from distributed_inference.model_manager.domain.model import Model, ModelId
from distributed_inference.model_manager.domain.model_version import (
    ModelVersion,
    ModelVersionId,
    ProfiledModelVersion,
)
from distributed_inference.model_manager.domain.model_version_graph import (
    LayerKey,
)
from distributed_inference.model_manager.domain.sub_model import SubModel, SubModelId
from distributed_inference.model_metadata_store.application.ports.outbound.model_metadata_store import (
    ModelMetadataStore,
)
from distributed_inference.model_profiler.application.ports.inbound.model_profiler import (
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
    async def register_model(self, model: Model) -> ModelId:
        return await self._model_metadata_store.register_model(model)

    @override
    async def upload_model_version(
        self, model_version: ModelVersion, bundle: ReadableArtifactBundle
    ) -> ModelVersionId:
        model_version_id = await self._model_metadata_store.register_model_version(
            model_version
        )
        artifact_key = ModelVersionArtifactKey(id=model_version_id)
        await self._artifact_store.put_artifact(artifact_key, bundle)

        model = await self._model_metadata_store.get_model(model_version.model_id)
        model_info = model.model_info

        async with self._artifact_materializer.materialize_artifact(
            artifact_key
        ) as materialized_artifact:
            profiled_model_version = await self._model_profiler.profile_model_version(
                materialized_artifact, model_info, model_version
            )
            await self._model_metadata_store.register_profiled_model_version(
                profiled_model_version
            )
        return model_version_id

    @override
    async def generate_sub_model(
        self, model_version_id: ModelVersionId, layers: Iterable[LayerKey]
    ) -> SubModel:

        SubModelId.check_valid_layers_format(layers)

        layers = tuple(layers)
        profiled_model_version = (
            await self._model_metadata_store.get_profiled_model_version(
                model_version_id
            )
        )
        if profiled_model_version is None:
            raise ValueError(
                f"Model graph for model version {model_version_id} still not ready"
            )

        sub_model_id = SubModelId(model_version_id=model_version_id, layers=layers)

        sub_model = SubModel(sub_model_id=sub_model_id)

        sub_model_id = await self._model_metadata_store.register_sub_model(sub_model)

        async with aiofiles.tempfile.TemporaryDirectory() as tmp_dir:
            split_artifact_paths = ArtifactWorkspace(
                root_path=Path(tmp_dir), entrypoint_path=None
            )

            artifact_key = ModelVersionArtifactKey(id=model_version_id)
            async with self._artifact_materializer.materialize_artifact(
                artifact_key
            ) as model_paths:
                await self._model_splitter.split_model(
                    profiled_model_version.model_version_graph,
                    layers,
                    model_paths,
                    split_artifact_paths,
                )

            sub_model_artifact_key = SubModelArtifactKey(id=sub_model_id)
            artifact_bundle = build_local_artifact_bundle_from_artifact_workspace(
                split_artifact_paths
            )
            await self._artifact_store.put_artifact(
                sub_model_artifact_key, artifact_bundle
            )

        return sub_model

    @override
    async def get_profiled_model_version(
        self, model_version_id: ModelVersionId
    ) -> ProfiledModelVersion:
        profiled_model_version = (
            await self._model_metadata_store.get_profiled_model_version(
                model_version_id
            )
        )
        if profiled_model_version is None:
            raise ValueError(
                f"Model graph for model version {model_version_id} still not ready"
            )
        return profiled_model_version
