from contextlib import AbstractContextManager
from pathlib import Path
from typing import Tuple, override

from distributed_inference.adapters.outbound.model_artifact.store.local import (
    local_storage_bundle_utils,
)
from distributed_inference.application.model_artifact.contracts.store.model_version_artifact_store import (
    ModelVersionArtifactStore,
)
from distributed_inference.application.model_artifact.domain.artifact_bundle import (
    ArtifactBundle,
    MaterializedArtifact,
)
from distributed_inference.domain.identifiers import (
    ModelVersionId,
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
    def put_model_version(
        self,
        model_version_id: ModelVersionId,
        bundle: ArtifactBundle,
    ) -> None:
        bundle_root_path, lock_file_path = self._build_bundle_root_path_and_lock_file(
            model_version_id
        )

        local_storage_bundle_utils.put_bundle(bundle, bundle_root_path, lock_file_path)
        pass

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
    def check_model_version_existance(
        self,
        model_version_id: ModelVersionId,
    ) -> bool:
        bundle_root_path, lock_path = self._build_bundle_root_path_and_lock_file(
            model_version_id
        )
        return local_storage_bundle_utils.check_bundle(bundle_root_path, lock_path)

    def get_model_version_bundle_path(
        self, model_version_id: ModelVersionId
    ) -> AbstractContextManager[MaterializedArtifact]:
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
            str(model_id.user_id), model_id.model_name
        )
        lock_path = self.lock_dir.joinpath(
            f"{model_id.user_id}_{model_id.model_name}_{model_version_id.version_number}.lock"
        )
        return root_path, lock_path
