from typing import Iterable


from distributed_inference.domain.identifiers import (
    ModelId,
    ModelVersionId,
    UserId,
    SubModelId,
)
from distributed_inference.domain.model_graph_info import (
    ModelInfo,
    ModelGraph,
    LayerKey,
)

from abc import ABC, abstractmethod


class ModelMetadataStore(ABC):
    @abstractmethod
    def register_model(
        self,
        owner_id: UserId,
    ) -> ModelId: ...

    @abstractmethod
    def register_model_version(
        self,
        model_id: ModelId,
        model_info: ModelInfo,
    ) -> ModelVersionId: ...

    @abstractmethod
    def register_model_version_graph(
        self,
        model_version_id: ModelVersionId,
        model_graph: ModelGraph,
    ) -> None: ...

    @abstractmethod
    def register_sub_model(
        self,
        model_version_id: ModelVersionId,
        layers: Iterable[LayerKey],
    ) -> SubModelId: ...

    @abstractmethod
    def get_model_graph(self, model_version_id: ModelVersionId) -> ModelGraph: ...

    @abstractmethod
    def get_model_info(self, model_version_id: ModelVersionId) -> ModelInfo: ...

    @abstractmethod
    def check_model_existence(self, model_id: ModelId) -> bool: ...

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
