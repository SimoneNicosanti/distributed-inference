from contextlib import AbstractContextManager
from pathlib import Path
from typing import Tuple, override

from distributed_inference.domain.identifiers import (
    ModelVersionId,
)
from distributed_inference.model_artifact.adapters.outbound.store.local import (
    local_storage_bundle_utils,
)
from distributed_inference.model_artifact.application.ports.outbound.store.model_version_artifact_store import (
    ModelVersionArtifactStore,
)
from distributed_inference.model_artifact.domain.artifact_bundle import (
    ArtifactBundle,
    ArtifactConcretePaths,
)


class LocalModelVersionArtifactStore(ModelVersionArtifactStore):
    def __init__(self, base_path: Path):
        self.base_path = base_path

        self.model_versions_dir = base_path.joinpath("model_versions")
        self.lock_dir = self.model_versions_dir.joinpath(".lock")

        base_path.mkdir(parents=True, exist_ok=True)
        self.model_versions_dir.mkdir(parents=True, exist_ok=True)
        self.lock_dir.mkdir(parents=True, exist_ok=True)

    @override
    async def put_model_version(
        self,
        model_version_id: ModelVersionId,
        bundle: ArtifactBundle,
    ) -> None:
        bundle_root_path, lock_file_path = self._build_bundle_root_path_and_lock_file(
            model_version_id
        )

        await local_storage_bundle_utils.put_bundle(
            bundle, bundle_root_path, lock_file_path
        )

    @override
    def get_model_version(
        self,
        model_version_id: ModelVersionId,
    ) -> AbstractContextManager[ArtifactBundle]:

        bundle_root_path, lock_path = self._build_bundle_root_path_and_lock_file(
            model_version_id
        )

        return local_storage_bundle_utils.get_bundle(bundle_root_path, lock_path)

    @override
    async def check_model_version_existence(
        self,
        model_version_id: ModelVersionId,
    ) -> bool:
        bundle_root_path, lock_path = self._build_bundle_root_path_and_lock_file(
            model_version_id
        )
        return await local_storage_bundle_utils.check_bundle(
            bundle_root_path, lock_path
        )

    def get_model_version_bundle_path(
        self, model_version_id: ModelVersionId
    ) -> AbstractContextManager[ArtifactConcretePaths]:
        bundle_root_path, lock_path = self._build_bundle_root_path_and_lock_file(
            model_version_id
        )

        return local_storage_bundle_utils.get_bundle_materialized_artifact(
            bundle_root_path, lock_path
        )

    def _build_bundle_root_path_and_lock_file(
        self, model_version_id: ModelVersionId
    ) -> Tuple[Path, Path]:
        model_id = model_version_id.model_id

        root_path = self.model_versions_dir.joinpath(
            str(model_id.user_id.user_id),
            model_id.model_name,
            str(model_version_id.version_number),
        )
        lock_path = self.lock_dir.joinpath(
            f"{model_id.user_id}_{model_id.model_name}_{model_version_id.version_number}.lock"
        )
        return root_path, lock_path
