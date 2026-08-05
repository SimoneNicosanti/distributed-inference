from abc import ABC, abstractmethod
from contextlib import AbstractAsyncContextManager
from typing import Iterable

from distributed_inference.artifact_store.domain.readable_artifact_bundle import (
    ReadableArtifactBundle,
)
from distributed_inference.model_manager.domain.model import Model, ModelId
from distributed_inference.model_manager.domain.model_version import (
    ModelVersion,
    ModelVersionId,
    ProfiledModelVersion,
)
from distributed_inference.model_manager.domain.model_version_graph import (
    LayerKey,
)
from distributed_inference.model_manager.domain.sub_model import SubModel, SubModelId


class ModelManager(ABC):
    @abstractmethod
    async def register_model(self, model: Model) -> ModelId: ...

    @abstractmethod
    async def upload_model_version(
        self,
        model_version: ModelVersion,
        bundle: ReadableArtifactBundle,
    ) -> ModelVersionId: ...

    @abstractmethod
    async def generate_sub_model(
        self,
        model_version_id: ModelVersionId,
        layers: Iterable[LayerKey],
    ) -> SubModel: ...

    @abstractmethod
    def download_sub_model(
        self,
        sub_model_id: SubModelId,
    ) -> AbstractAsyncContextManager[ReadableArtifactBundle]: ...

    @abstractmethod
    async def get_profiled_model_version(
        self, model_version_id: ModelVersionId
    ) -> ProfiledModelVersion: ...
