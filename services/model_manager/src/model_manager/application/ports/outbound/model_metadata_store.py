from abc import ABC, abstractmethod

from model_manager.domain.model import Model
from model_manager.domain.model_partition import ModelPartition
from model_manager.domain.profiled_model_variant import ProfiledModelVariant
from shared.model.model import (
    ModelId,
)
from shared.model.model_variant import (
    ModelVariantId,
)


class ModelMetadataStore(ABC):
    ## Model APIs
    @abstractmethod
    async def add_model(self, model: Model) -> None: ...

    @abstractmethod
    async def get_model(self, model_id: ModelId) -> Model: ...

    @abstractmethod
    async def check_model_existence(self, model_id: ModelId) -> bool: ...

    ## Profiled Model Variant APIs
    @abstractmethod
    async def add_profiled_model_variant(
        self,
        profiled_model_variant: ProfiledModelVariant,
    ) -> None: ...

    @abstractmethod
    async def get_profiled_model_variant(
        self, model_variant_id: ModelVariantId
    ) -> ProfiledModelVariant: ...

    @abstractmethod
    async def check_profiled_model_variant_existence(
        self,
        model_variant_id: ModelVariantId,
    ) -> bool: ...

    ## Model Partition APIs
    @abstractmethod
    async def add_model_partition(
        self,
        model_partition: ModelPartition,
    ) -> None: ...
