from typing import BinaryIO

from distributed_inference.domain.identifiers import (
    SubModelId,
)
from contextlib import AbstractContextManager

from abc import ABC, abstractmethod


class SubModelArtifactStore(ABC):
    @abstractmethod
    def check_sub_model_existance(
        self,
        artifact_id: SubModelId,
    ) -> bool: ...

    @abstractmethod
    def put_sub_model(
        self,
        sub_model_id: SubModelId,
        binary_io: BinaryIO,
    ) -> None: ...

    @abstractmethod
    def get_sub_model(
        self,
        sub_model_id: SubModelId,
    ) -> AbstractContextManager[BinaryIO]: ...
