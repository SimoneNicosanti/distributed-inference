from abc import ABC, abstractmethod

from shared.model.model_version import ModelVersionId


class ModelMetadataStore(ABC):
    @abstractmethod
    async def get_all_model_version_ids(self) -> list[ModelVersionId]: ...
