from abc import ABC, abstractmethod
from contextlib import AbstractContextManager

from distributed_inference.application.model_artifact.domain.artifact_bundle import (
    MaterializedArtifact,
)
from distributed_inference.domain.identifiers import (
    SubModelId,
)


class SubModelMaterializer(ABC):
    @abstractmethod
    def materialize_sub_model(
        self,
        sub_model_id: SubModelId,
    ) -> AbstractContextManager[MaterializedArtifact]: ...
