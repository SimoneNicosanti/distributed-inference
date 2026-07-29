from abc import ABC, abstractmethod
from contextlib import AbstractContextManager

from distributed_inference.domain.identifiers import (
    SubModelId,
)
from distributed_inference.model_artifact.domain.artifact_bundle import (
    ArtifactConcretePaths,
)


class SubModelMaterializer(ABC):
    @abstractmethod
    def materialize_sub_model(
        self,
        sub_model_id: SubModelId,
    ) -> AbstractContextManager[ArtifactConcretePaths]: ...
