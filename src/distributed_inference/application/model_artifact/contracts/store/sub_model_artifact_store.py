from abc import ABC, abstractmethod
from contextlib import AbstractContextManager

from distributed_inference.application.model_artifact.domain.artifact_bundle import (
    ArtifactBundle,
)
from distributed_inference.domain.identifiers import (
    SubModelId,
)


class SubModelArtifactStore(ABC):
    @abstractmethod
    def put_sub_model(
        self,
        sub_model_id: SubModelId,
        bundle: ArtifactBundle,
    ) -> None: ...

    @abstractmethod
    def get_sub_model(
        self,
        sub_model_id: SubModelId,
    ) -> AbstractContextManager[ArtifactBundle]: ...

    @abstractmethod
    def check_sub_model_existance(
        self,
        artifact_id: SubModelId,
    ) -> bool: ...
