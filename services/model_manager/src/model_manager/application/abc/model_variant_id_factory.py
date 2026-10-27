from abc import ABC, abstractmethod

from model_manager.domain.model_variant import ModelVariantInfo
from shared.model.model import ModelId
from shared.model.model_variant import ModelVariantId


class ModelVariantIdFactory(ABC):
    @staticmethod
    @abstractmethod
    async def build_model_variant_id(
        model_id: ModelId, model_variant_info: ModelVariantInfo
    ) -> ModelVariantId: ...
