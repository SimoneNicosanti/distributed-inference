from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator

from shared.model.model_version import ModelVersionId


class PendingModelsPoller(ABC):
    @abstractmethod
    def pending_models(
        self,
    ) -> AsyncGenerator[ModelVersionId]: ...
