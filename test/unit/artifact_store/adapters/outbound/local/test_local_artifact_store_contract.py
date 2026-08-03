from pathlib import Path

import pytest
from typing_extensions import override

from distributed_inference.artifact_store.adapters.outbound.local.local_artifact_store import (
    LocalArtifactStore,
)
from distributed_inference.artifact_store.application.ports.outbound.artifact_store import (
    ArtifactStore,
)
from test.contracts.artifact_store.artifact_store_contract import (
    ArtifactStoreContract,
)

pytestmark = [pytest.mark.unit, pytest.mark.contract]

class TestLocalArtifactStoreContract(ArtifactStoreContract):
    @override
    def build_store(self, base_path: Path) -> ArtifactStore:
        return LocalArtifactStore(base_path)
