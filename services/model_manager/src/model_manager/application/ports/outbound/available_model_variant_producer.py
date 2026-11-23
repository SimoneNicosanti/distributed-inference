from abc import ABC, abstractmethod

from shared.model.model_variant import ModelVariantId


class AvailableModelVariantProducer(ABC):
    @abstractmethod
    async def publish_available_model_variant(
        self, model_variant_id: ModelVariantId
    ) -> None: ...
