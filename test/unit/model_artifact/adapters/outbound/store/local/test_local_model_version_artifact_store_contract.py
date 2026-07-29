from pathlib import Path
from typing import override

from distributed_inference.model_artifact.adapters.outbound.store.local.local_model_version_artifact_store import (
    LocalModelVersionArtifactStore,
)
from distributed_inference.model_artifact.application.ports.outbound.store.model_version_artifact_store import (
    ModelVersionArtifactStore,
)
from test.contracts.model_artifact.model_version_artifact_store_contract import (
    ModelVersionArtifactStoreContract,
)


class TestLocalModelVersionArtifactStoreContract(ModelVersionArtifactStoreContract):
    @override
    def build_store(
        self,
        base_path: Path,
    ) -> ModelVersionArtifactStore:
        return LocalModelVersionArtifactStore(base_path)
