from collections.abc import Iterable
from pathlib import Path
from typing import override

import aiofiles

from artifacts.contracts.artifact_bundle import ArtifactBundle
from artifacts.contracts.artifact_workspace import ArtifactWorkspace
from artifacts.storage.artifact_store import ArtifactStore
from model_manager.application.abc.artifact_ref_factory import ArtifactRefFactory
from model_manager.application.ports.inbound.model_partitioner import ModelPartitioner
from model_manager.application.ports.inbound.model_registry import ModelRegistry
from model_manager.application.ports.inbound.model_variant_registry import (
    ModelVariantRegistry,
)
from model_manager.application.ports.outbound.model_metadata_store import (
    ModelMetadataStore,
)
from model_manager.application.ports.outbound.model_partition_extractor import (
    ModelPartitionExtractor,
)
from model_manager.application.ports.outbound.model_profiler import (
    ModelProfiler,
)
from model_manager.domain.model import Model
from model_manager.domain.model_partition import ModelPartition
from model_manager.domain.model_variant import ModelVariant
from model_manager.domain.profiled_model_variant import ProfiledModelVariant
from shared.artifact.artifact_ref import ArtifactRef
from shared.model.keys import LayerKey
from shared.model.model_partition import ModelPartitionId
from shared.model.model_variant import ModelVariantId


class DefaultModelManager(ModelRegistry, ModelVariantRegistry, ModelPartitioner):
    def __init__(
        self,
        model_profiler: ModelProfiler,
        artifact_store: ArtifactStore,
        artifact_ref_factory: ArtifactRefFactory,
        model_metadata_store: ModelMetadataStore,
        model_splitter: ModelPartitionExtractor,
    ):
        self._model_profiler = model_profiler

        self._artifact_store = artifact_store
        self._artifact_ref_factory = artifact_ref_factory
        self._model_metadata_store = model_metadata_store

        self._model_splitter = model_splitter

        self._pending_model_variants: dict[
            ModelVariantId, tuple[ModelVariant, ArtifactRef]
        ] = {}

    @override
    async def register_model(self, model: Model) -> None:
        await self._model_metadata_store.add_model(model)

    @override
    async def start_model_variant_registration(
        self, model_variant: ModelVariant
    ) -> ArtifactRef:

        artifact_ref = await self._artifact_ref_factory.build_artifact_ref(
            model_variant.model_variant_id.model_dump_json()
        )
        self._pending_model_variants[model_variant.model_variant_id] = (
            model_variant,
            artifact_ref,
        )

        return artifact_ref

    @override
    async def finalize_model_variant_registration(
        self, model_variant_id: ModelVariantId
    ) -> None:

        registration_info = self._pending_model_variants.get(model_variant_id, None)

        if registration_info is None:
            raise ValueError(
                f"Model variant registration not started for {model_variant_id}"
            )

        model_variant, artifact_ref = registration_info

        model = await self._model_metadata_store.get_model(model_variant.model_id)

        async with self._artifact_store.download_artifact(
            artifact_ref
        ) as artifact_bundle:
            artifact_workspace = artifact_bundle.to_workspace()

            profile = await self._model_profiler.profile_model_variant(
                artifact_workspace=artifact_workspace,
                model_info=model.model_info,
                model_variant=model_variant,
            )

        profiled_model_variant = ProfiledModelVariant(
            model_variant=model_variant,
            profile=profile,
            artifact_ref=artifact_ref,
        )
        await self._model_metadata_store.add_profiled_model_variant(
            profiled_model_variant
        )

        self._pending_model_variants.pop(model_variant_id, None)

    @override
    async def generate_model_variant_partition(
        self,
        model_variant_id: ModelVariantId,
        layers: Iterable[LayerKey],
    ) -> ModelPartition:

        ModelPartitionId.check_valid_layers_format(layers)

        layers = tuple(layers)
        profiled_model_version = (
            await self._model_metadata_store.get_profiled_model_variant(
                model_variant_id
            )
        )

        partition_id = ModelPartitionId(
            model_variant_id=model_variant_id, layers=layers
        )

        async with aiofiles.tempfile.TemporaryDirectory() as tmp_dir:
            partition_workspace = ArtifactWorkspace(
                root_path=Path(tmp_dir), entrypoint_path=None
            )

            variant_artifact_ref = profiled_model_version.artifact_ref
            async with self._artifact_store.download_artifact(
                variant_artifact_ref
            ) as variant_bundle:
                model_variant_workspace = variant_bundle.to_workspace()

                await self._model_splitter.extract_model_partition(
                    profiled_model_version.profile.model_variant_graph,
                    layers,
                    model_variant_workspace,
                    partition_workspace,
                )

            partition_artifact_ref = (
                await self._artifact_ref_factory.build_artifact_ref(
                    partition_id.model_dump_json()
                )
            )
            partition_bundle = ArtifactBundle.from_workspace(partition_workspace)
            await self._artifact_store.put_artifact(
                partition_artifact_ref, partition_bundle
            )

        partition = ModelPartition(
            model_partition_id=partition_id, artifact_ref=partition_artifact_ref
        )

        await self._model_metadata_store.add_model_partition(partition)

        return partition
