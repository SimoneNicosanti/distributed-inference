from collections.abc import Iterable
from pathlib import Path
from typing import override

import aiofiles

from artifacts.contracts.artifact_ref import build_artifact_ref
from artifacts.storage.artifact_store import ArtifactStore
from artifacts.workspace.artifact_bundle import ArtifactBundle
from artifacts.workspace.artifact_workspace import (
    ArtifactWorkspace,
    build_artifact_bundle_from_artifact_workspace,
    build_artifact_workspace_from_artifact_bundle,
)
from model_manager.application.ports.inbound.model_manager import (
    ModelManager,
)
from model_manager.application.ports.outbound.model_metadata_store import (
    ModelMetadataStore,
)
from model_manager.application.ports.outbound.model_profiler import (
    ModelProfiler,
)
from model_manager.application.ports.outbound.model_splitter import (
    ModelSplitter,
)
from model_manager.domain.profiled_model_version import ProfiledModelVersion
from shared.model.keys import LayerKey
from shared.model.model import Model, ModelId
from shared.model.model_version import (
    ModelVersion,
    ModelVersionId,
)
from shared.model.sub_model import (
    SubModel,
    SubModelId,
)


class DefaultModelManager(ModelManager):
    ## TODO: We should add rollback logick with rollback library

    def __init__(
        self,
        model_profiler: ModelProfiler,
        artifact_store: ArtifactStore,
        model_metadata_store: ModelMetadataStore,
        model_splitter: ModelSplitter,
    ):
        self._model_profiler = model_profiler

        self._artifact_store = artifact_store
        self._model_metadata_store = model_metadata_store

        self._model_splitter = model_splitter

    @override
    async def register_model(self, model: Model) -> ModelId:
        return await self._model_metadata_store.register_model(model)

    @override
    async def upload_model_version(
        self, model_version: ModelVersion, bundle: ArtifactBundle
    ) -> ModelVersionId:
        model_version_id = await self._model_metadata_store.register_model_version(
            model_version
        )

        artifact_ref = await build_artifact_ref(model_version_id.model_dump_json())
        await self._artifact_store.put_artifact(artifact_ref, bundle)

        model = await self._model_metadata_store.get_model(model_version.model_id)
        model_info = model.model_info

        artifact_workspace = build_artifact_workspace_from_artifact_bundle(bundle)
        profiled_model_version = await self._model_profiler.profile_model_version(
            artifact_workspace, model_info, model_version
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
            sub_model_workspace = ArtifactWorkspace(
                root_path=Path(tmp_dir), entrypoint_path=None
            )

            model_ver_artifact_ref = await build_artifact_ref(
                sub_model_id.model_version_id.model_dump_json()
            )
            async with self._artifact_store.download_artifact(
                model_ver_artifact_ref
            ) as model_bundle:
                model_ver_workspace = build_artifact_workspace_from_artifact_bundle(
                    model_bundle
                )

                await self._model_splitter.split_model(
                    profiled_model_version.model_version_graph,
                    layers,
                    model_ver_workspace,
                    sub_model_workspace,
                )

            sub_model_artifact_ref = await build_artifact_ref(
                sub_model_id.model_dump_json()
            )
            sub_model_bundle = build_artifact_bundle_from_artifact_workspace(
                sub_model_workspace
            )
            await self._artifact_store.put_artifact(
                sub_model_artifact_ref, sub_model_bundle
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
