from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from pathlib import Path

from distributed_inference.domain.identifiers import (
    SubModelId,
)


class SubModelMaterializer(ABC):
    @abstractmethod
    def materialize_sub_model(
        self,
        sub_model_id: SubModelId,
    ) -> AbstractContextManager[Path]: ...
