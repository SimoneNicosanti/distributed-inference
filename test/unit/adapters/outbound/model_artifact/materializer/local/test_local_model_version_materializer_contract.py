from pathlib import Path
from typing import override

from distributed_inference.adapters.outbound.model_artifact.materializer.local.local_model_version_materializer import (
    LocalModelVersionMaterializer,
)
from distributed_inference.adapters.outbound.model_artifact.store.local.local_model_version_artifact_store import (
    LocalModelVersionArtifactStore,
)
from distributed_inference.application.model_artifact.contracts.materializer.model_version_materializer import (
    ModelVersionMaterializer,
)
from distributed_inference.application.model_artifact.contracts.store.model_version_artifact_store import (
    ModelVersionArtifactStore,
)
from test.contracts.model_version_materializer_contract import (
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
