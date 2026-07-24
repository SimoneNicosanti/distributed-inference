from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from typing import BinaryIO

from distributed_inference.domain.identifiers import (
    ModelVersionId,
)


class ModelVersionArtifactStore(ABC):
    @abstractmethod
    def put_model_version(
        self,
        model_version_id: ModelVersionId,
        binary_io: BinaryIO,
    ) -> None: ...

    @abstractmethod
    def get_model_version(
        self,
        model_version_id: ModelVersionId,
    ) -> AbstractContextManager[BinaryIO]: ...

    @abstractmethod
    def check_model_version_existance(
        self,
        artifact_id: ModelVersionId,
    ) -> bool: ...
