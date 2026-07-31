from abc import ABC, abstractmethod
from contextlib import AbstractContextManager

from distributed_inference.domain.identifiers import (
    SubModelId,
)
from distributed_inference.model_artifact.domain.artifact_bundle import (
    ArtifactBundle,
)


class SubModelArtifactStore(ABC):
    @abstractmethod
    async def put_sub_model(
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
    async def check_sub_model_existence(
        self,
        sub_model_id: SubModelId,
    ) -> bool: ...
