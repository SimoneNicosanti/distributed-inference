from abc import ABC, abstractmethod
from contextlib import AbstractAsyncContextManager
from typing import Iterable

from distributed_inference.artifact_store.domain.readable_artifact_bundle import (
    ReadableArtifactBundle,
)
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
    async def register_model(
        self,
        owner_id: UserId,
        model_name: str,
    ) -> ModelId: ...

    @abstractmethod
    async def put_model_version(
        self,
        model_id: ModelId,
        model_info: ModelInfo,
        bundle: ReadableArtifactBundle,
    ) -> ModelVersionId: ...

    @abstractmethod
    async def generate_sub_model(
        self,
        model_version_id: ModelVersionId,
        layers: Iterable[LayerKey],
    ) -> SubModelId: ...

    @abstractmethod
    def get_sub_model(
        self,
        sub_model_id: SubModelId,
    ) -> AbstractAsyncContextManager[ReadableArtifactBundle]: ...

    @abstractmethod
    async def get_model_graph(self, model_version_id: ModelVersionId) -> ModelGraph: ...

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
