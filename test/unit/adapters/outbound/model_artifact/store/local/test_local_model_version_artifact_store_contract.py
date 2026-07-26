from pathlib import Path
from typing import override

from distributed_inference.adapters.outbound.model_artifact.store.local.local_model_version_artifact_store import (
    LocalModelVersionArtifactStore,
)
from distributed_inference.application.model_artifact.contracts.store.model_version_artifact_store import (
    ModelVersionArtifactStore,
)
from test.contracts.model_version_artifact_store_contract import (
    ModelVersionArtifactStoreContract,
)


class TestLocalModelVersionArtifactStoreContract(ModelVersionArtifactStoreContract):
    @override
    def build_store(
        self,
        base_path: Path,
    ) -> ModelVersionArtifactStore:
        return LocalModelVersionArtifactStore(base_path)
