from contextlib import AbstractContextManager
from pathlib import Path
from typing import Tuple, override

from distributed_inference.adapters.outbound.model_artifact.store.local import (
    local_storage_bundle_utils,
)
from distributed_inference.application.model_artifact.contracts.store.sub_model_artifact_store import (
    SubModelArtifactStore,
)
from distributed_inference.application.model_artifact.domain.artifact_bundle import (
    ArtifactBundle,
    ArtifactConcretePaths,
)
from distributed_inference.domain.identifiers import (
    SubModelId,
)
from distributed_inference.domain.model_graph_info import LayerKey


class LocalSubModelArtifactStore(SubModelArtifactStore):
    def __init__(
        self,
        base_path: Path,
    ):
        self.base_path = base_path

        self.sub_models_dir = base_path.joinpath("sub_models")
        self.lock_dir = self.sub_models_dir.joinpath(".lock")

        base_path.mkdir(parents=True, exist_ok=True)
        self.sub_models_dir.mkdir(parents=True, exist_ok=True)
        self.lock_dir.mkdir(parents=True, exist_ok=True)

    @override
    def put_sub_model(
        self,
        sub_model_id: SubModelId,
        bundle: ArtifactBundle,
    ) -> None:
        bundle_root_path, lock_path = self._build_bundle_root_path_and_lock_file(
            sub_model_id
        )

        local_storage_bundle_utils.put_bundle(bundle, bundle_root_path, lock_path)

    @override
    def get_sub_model(
        self,
        sub_model_id: SubModelId,
    ) -> AbstractContextManager[ArtifactBundle]:

        bundle_root_path, lock_path = self._build_bundle_root_path_and_lock_file(
            sub_model_id
        )

        return local_storage_bundle_utils.get_bundle(bundle_root_path, lock_path)

    def get_sub_model_path(
        self, sub_model_id: SubModelId
    ) -> AbstractContextManager[ArtifactConcretePaths]:
        bundle_root_path, lock_path = self._build_bundle_root_path_and_lock_file(
            sub_model_id
        )

        return local_storage_bundle_utils.get_bundle_materialized_artifact(
            bundle_root_path, lock_path
        )

    def _build_bundle_root_path_and_lock_file(
        self, sub_model_id: SubModelId
    ) -> Tuple[Path, Path]:

        model_id = sub_model_id.model_version_id.model_id

        layers_hash = self._hash_layers(sub_model_id.layers)

        bundle_root_path = self.sub_models_dir.joinpath(
            str(model_id.user_id.user_id),
            model_id.model_name,
            str(sub_model_id.model_version_id.version_number),
            layers_hash,
        )

        lock_path = self.lock_dir.joinpath(
            f"{model_id.user_id}_{model_id.model_name}_{sub_model_id.model_version_id.version_number}_{layers_hash}.lock"
        )

        return bundle_root_path, lock_path

    def _hash_layers(self, layers: Tuple[LayerKey, ...]) -> str:
        from hashlib import md5

        payload = "\0".join(str(layer) for layer in layers)
        return md5(payload.encode("utf-8")).hexdigest()

    @override
    def check_sub_model_existence(self, sub_model_id: SubModelId) -> bool:
        bundle_root_path, lock_path = self._build_bundle_root_path_and_lock_file(
            sub_model_id
        )
        return local_storage_bundle_utils.check_bundle(bundle_root_path, lock_path)
