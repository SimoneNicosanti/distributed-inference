from distributed_inference.domain.identifiers import (
    ModelVersionId,
)
from contextlib import AbstractContextManager

from pathlib import Path

from abc import ABC, abstractmethod


class SubModelMaterializer(ABC):
    @abstractmethod
    def materialize_sub_model(
        self,
        sub_model_id: ModelVersionId,
    ) -> AbstractContextManager[Path]: ...
