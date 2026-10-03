from abc import ABC, abstractmethod
from collections.abc import Iterable

from model_manager.domain.model_partition import ModelPartition
from shared.model.keys import LayerKey
from shared.model.model_variant import ModelVariantId


class ModelPartitioner(ABC):
    @abstractmethod
    async def generate_model_variant_partition(
        self,
        model_variant_id: ModelVariantId,
        layers: Iterable[LayerKey],
    ) -> ModelPartition: ...
