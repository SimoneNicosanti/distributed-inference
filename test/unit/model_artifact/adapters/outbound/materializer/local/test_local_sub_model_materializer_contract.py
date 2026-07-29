from pathlib import Path
from typing import override

from distributed_inference.model_artifact.adapters.outbound.materializer.local.local_sub_model_materializer import (
    LocalSubModelMaterializer,
)
from distributed_inference.model_artifact.adapters.outbound.store.local.local_sub_model_artifact_store import (
    LocalSubModelArtifactStore,
)
from distributed_inference.model_artifact.application.ports.outbound.materializer.sub_model_materializer import (
    SubModelMaterializer,
)
from distributed_inference.model_artifact.application.ports.outbound.store.sub_model_artifact_store import (
    SubModelArtifactStore,
)
from test.contracts.model_artifact.sub_model_materializer_contract import (
    SubModelMaterializerContract,
)


class TestLocalSubModelMaterializerContract(SubModelMaterializerContract):
    @override
    def build_dependencies(
        self,
        base_path: Path,
    ) -> tuple[
        SubModelMaterializer,
        SubModelArtifactStore,
    ]:
        store = LocalSubModelArtifactStore(base_path)

        return (
            LocalSubModelMaterializer(store),
            store,
        )
