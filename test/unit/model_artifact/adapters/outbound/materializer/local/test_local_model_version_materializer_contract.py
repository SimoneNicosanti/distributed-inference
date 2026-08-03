from pathlib import Path
from typing import override

from distributed_inference.model_materializer.adapters.outbound.local.local_model_version_materializer import (
    LocalModelVersionMaterializer,
)
from distributed_inference.model_artifact.adapters.outbound.local.local_model_version_artifact_store import (
    LocalModelVersionArtifactStore,
)
from distributed_inference.model_materializer.application.ports.outbound.model_version_materializer import (
    ModelVersionMaterializer,
)
from distributed_inference.model_artifact.application.ports.outbound.model_version_artifact_store import (
    ModelVersionArtifactStore,
)
from test.contracts.model_artifact.model_version_materializer_contract import (
    ModelVersionMaterializerContract,
)


class TestLocalModelVersionMaterializerContract(ModelVersionMaterializerContract):
    @override
    def build_dependencies(
        self,
        base_path: Path,
    ) -> tuple[
        ModelVersionMaterializer,
        ModelVersionArtifactStore,
    ]:
        store = LocalModelVersionArtifactStore(base_path)

        return (
            LocalModelVersionMaterializer(store),
            store,
        )
