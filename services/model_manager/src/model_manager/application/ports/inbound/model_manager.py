from abc import ABC, abstractmethod
from collections.abc import Iterable

from artifacts.workspace.artifact_bundle import ArtifactBundle
from integration.model.keys import LayerKey
from integration.model.model import Model, ModelId
from integration.model.model_version import (
    ModelVersion,
    ModelVersionId,
)
from integration.model.sub_model import SubModel
from model_manager.domain.profiled_model_version import ProfiledModelVersion


class ModelManager(ABC):
    @abstractmethod
    async def register_model(self, model: Model) -> ModelId: ...

    @abstractmethod
    async def upload_model_version(
        self,
        model_version: ModelVersion,
        bundle: ArtifactBundle,
    ) -> ModelVersionId: ...

    @abstractmethod
    async def generate_sub_model(
        self,
        model_version_id: ModelVersionId,
        layers: Iterable[LayerKey],
    ) -> SubModel: ...

    @abstractmethod
    async def get_profiled_model_version(
        self, model_version_id: ModelVersionId
    ) -> ProfiledModelVersion: ...
