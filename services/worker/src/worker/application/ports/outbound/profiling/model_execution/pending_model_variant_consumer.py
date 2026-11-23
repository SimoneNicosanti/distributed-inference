from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator

from shared.model.model_variant import ModelVariantId


class PendingModelVariantConsumer(ABC):
    @abstractmethod
    def pending_model_variant_ids(self) -> AsyncGenerator[ModelVariantId]: ...

    @abstractmethod
    async def acknowledge(self, model_variant_id: ModelVariantId) -> None: ...
