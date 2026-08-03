from pathlib import Path

import pytest
from typing_extensions import override

from distributed_inference.artifact_materializer.adapters.outbound.local.local_artifact_materializer import (
    LocalArtifactMaterializer,
)
from distributed_inference.artifact_materializer.application.ports.outbound.artifact_materializer import (
    ArtifactMaterializer,
)
from distributed_inference.artifact_store.adapters.outbound.local.local_artifact_store import (
    LocalArtifactStore,
)
from distributed_inference.artifact_store.application.ports.outbound.artifact_store import (
    ArtifactStore,
)
from test.contracts.artifact_materializer.artifact_materializer_contract import (
    ArtifactMaterializerContract,
)

pytestmark = [pytest.mark.unit, pytest.mark.contract]

class TestLocalArtifactMaterializerContract(ArtifactMaterializerContract):
    @override
    def build_materializer_and_store(
        self,
        base_path: Path,
    ) -> tuple[ArtifactMaterializer, ArtifactStore]:
        store = LocalArtifactStore(base_path)
        return LocalArtifactMaterializer(store), store
