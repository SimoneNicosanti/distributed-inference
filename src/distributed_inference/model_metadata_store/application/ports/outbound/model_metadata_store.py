from abc import ABC, abstractmethod
from typing import Iterable

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


class ModelMetadataStore(ABC):
    @abstractmethod
    async def register_model(
        self,
        owner_id: UserId,
        model_name: str,
    ) -> ModelId: ...

    @abstractmethod
    async def register_model_version(
        self,
        model_id: ModelId,
        model_info: ModelInfo,
    ) -> ModelVersionId: ...

    @abstractmethod
    async def register_model_version_graph(
        self,
        model_version_id: ModelVersionId,
        model_graph: ModelGraph,
    ) -> None: ...

    @abstractmethod
    async def register_sub_model(
        self,
        model_version_id: ModelVersionId,
        layers: Iterable[LayerKey],
    ) -> SubModelId: ...

    @abstractmethod
    async def get_model_graph(
        self, model_version_id: ModelVersionId
    ) -> ModelGraph | None: ...

    @abstractmethod
    async def get_model_info(self, model_version_id: ModelVersionId) -> ModelInfo: ...

    @abstractmethod
    async def check_model_existence(self, model_id: ModelId) -> bool: ...

    @abstractmethod
    async def check_model_version_existence(
        self,
        model_version_id: ModelVersionId,
    ) -> bool: ...

    @abstractmethod
    async def check_sub_model_existence(
        self,
        sub_model_id: SubModelId,
    ) -> bool: ...
