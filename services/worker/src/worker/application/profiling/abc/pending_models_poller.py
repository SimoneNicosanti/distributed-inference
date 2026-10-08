from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator

from shared.model.model_variant import ModelVariantId


class PendingModelsPoller(ABC):
    @abstractmethod
    def pending_model_ids(
        self,
    ) -> AsyncGenerator[ModelVariantId]: ...
