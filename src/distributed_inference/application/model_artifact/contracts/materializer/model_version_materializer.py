from distributed_inference.domain.identifiers import (
    ModelVersionId,
)
from contextlib import AbstractContextManager

from pathlib import Path

from abc import ABC, abstractmethod


class ModelVersionMaterializer(ABC):
    @abstractmethod
    def materialize_model_version(
        self,
        model_version_id: ModelVersionId,
    ) -> AbstractContextManager[Path]: ...
