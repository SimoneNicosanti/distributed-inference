from typing import Iterable, BinaryIO
from contextlib import AbstractContextManager


from distributed_inference.domain.identifiers import (
    ModelId,
    ModelVersionId,
    UserId,
    SubModelId,
)
from distributed_inference.domain.model_graph_info import (
    ModelInfo,
    LayerKey,
    ModelGraph,
)

from abc import ABC, abstractmethod


class ModelManager(ABC):
    @abstractmethod
    def register_model(
        self,
        owner_id: UserId,
    ) -> ModelId: ...

    @abstractmethod
    def upload_model_version(
        self,
        model_id: ModelId,
        model_info: ModelInfo,
        binary_io: BinaryIO,
    ) -> ModelVersionId: ...

    @abstractmethod
    def generate_submodel(
        self,
        version_id: ModelVersionId,
        component_layers: Iterable[LayerKey],
    ) -> SubModelId: ...

    @abstractmethod
    def open_submodel(
        self,
        submodel_id: SubModelId,
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
