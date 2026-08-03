from pathlib import Path
from typing import override

from distributed_inference.model_artifact.adapters.outbound.local.local_sub_model_artifact_store import (
    LocalSubModelArtifactStore,
)
from distributed_inference.model_artifact.application.ports.outbound.sub_model_artifact_store import (
    SubModelArtifactStore,
)
from test.contracts.model_artifact.sub_model_artifact_store_contract import (
    SubModelArtifactStoreContract,
)


class TestLocalSubModelArtifactStoreContract(SubModelArtifactStoreContract):
    @override
    def build_store(
        self,
        base_path: Path,
    ) -> SubModelArtifactStore:
        return LocalSubModelArtifactStore(base_path)
