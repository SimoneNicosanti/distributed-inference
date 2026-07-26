from pathlib import Path
from typing import override

from distributed_inference.adapters.outbound.model_artifact.materializer.local.local_sub_model_materializer import (
    LocalSubModelMaterializer,
)
from distributed_inference.adapters.outbound.model_artifact.store.local.local_sub_model_artifact_store import (
    LocalSubModelArtifactStore,
)
from distributed_inference.application.model_artifact.contracts.materializer.sub_model_materializer import (
    SubModelMaterializer,
)
from distributed_inference.application.model_artifact.contracts.store.sub_model_artifact_store import (
    SubModelArtifactStore,
)
from test.contracts.sub_model_materializer_contract import (
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
