from pathlib import Path
from typing import override

from distributed_inference.adapters.outbound.model_artifact.store.local.local_sub_model_artifact_store import (
    LocalSubModelArtifactStore,
)
from distributed_inference.application.model_artifact.contracts.store.sub_model_artifact_store import (
    SubModelArtifactStore,
)
from test.contracts.sub_model_artifact_store_contract import (
    SubModelArtifactStoreContract,
)


class TestLocalSubModelArtifactStoreContract(SubModelArtifactStoreContract):
    @override
    def build_store(
        self,
        base_path: Path,
    ) -> SubModelArtifactStore:
        return LocalSubModelArtifactStore(base_path)
