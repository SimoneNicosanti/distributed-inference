from abc import ABC, abstractmethod

from shared.model.model_variant import ModelVariantId
from worker.domain.profiling.model_execution.model_static_profile import (
    ModelStaticProfile,
)


class ModelStaticProfileReader(ABC):
    @abstractmethod
    async def get_all_model_static_profile_ids(
        self,
    ) -> list[ModelVariantId]: ...

    @abstractmethod
    async def get_model_static_profile(
        self,
        model_variant_id: ModelVariantId,
    ) -> ModelStaticProfile: ...
