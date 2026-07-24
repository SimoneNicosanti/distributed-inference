from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from typing import BinaryIO, Iterable

from distributed_inference.domain.identifiers import (
    ModelId,
    ModelVersionId,
    SubModelId,
    UserId,
)
from distributed_inference.domain.model_graph_info import (
    LayerKey,
    ModelGraph,
    ModelInfo,
)


class ModelManager(ABC):
    @abstractmethod
    def register_model(
        self,
        owner_id: UserId,
        model_name: str,
    ) -> ModelId: ...

    @abstractmethod
    def upload_model_version(
        self,
        model_id: ModelId,
        model_info: ModelInfo,
        binary_io: BinaryIO,
    ) -> ModelVersionId: ...

    @abstractmethod
    def generate_sub_model(
        self,
        model_version_id: ModelVersionId,
        layers: Iterable[LayerKey],
    ) -> SubModelId: ...

    @abstractmethod
    def download_sub_model(
        self,
        sub_model_id: SubModelId,
    ) -> AbstractContextManager[BinaryIO]: ...

    @abstractmethod
    def get_model_graph(self, model_version_id: ModelVersionId) -> ModelGraph: ...

    @abstractmethod
    def check_model_version_existence(
        self,
        model_version_id: ModelVersionId,
    ) -> bool: ...

    @abstractmethod
    def check_sub_model_existence(
        self,
        sub_model_id: SubModelId,
    ) -> bool: ...
