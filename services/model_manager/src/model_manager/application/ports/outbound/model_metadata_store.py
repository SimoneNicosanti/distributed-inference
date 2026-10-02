from abc import ABC, abstractmethod

from model_manager.domain.profiled_model_version import ProfiledModelVersion
from shared.model.model import (
    Model,
    ModelId,
)
from shared.model.model_version import (
    ModelVersion,
    ModelVersionId,
)


class ModelMetadataStore(ABC):
    ## Model APIs
    @abstractmethod
    async def register_model(self, model: Model) -> ModelId: ...

    @abstractmethod
    async def get_model(self, model_id: ModelId) -> Model: ...

    @abstractmethod
    async def check_model_existence(self, model_id: ModelId) -> bool: ...

    ## Model Version APIs
    @abstractmethod
    async def register_model_version(
        self,
        model_version: ModelVersion,
    ) -> ModelVersionId: ...

    @abstractmethod
    async def get_model_version(
        self, model_version_id: ModelVersionId
    ) -> ModelVersion: ...

    @abstractmethod
    async def check_model_version_existence(
        self,
        model_version_id: ModelVersionId,
    ) -> bool: ...

    ## Profiled Model Version APIs
    @abstractmethod
    async def register_profiled_model_version(
        self,
        profiled_model_version: ProfiledModelVersion,
    ) -> ModelVersionId: ...

    @abstractmethod
    async def get_profiled_model_version(
        self, model_version_id: ModelVersionId
    ) -> ProfiledModelVersion | None: ...

    @abstractmethod
    async def check_profiled_model_version_existence(
        self,
        model_version_id: ModelVersionId,
    ) -> bool: ...
