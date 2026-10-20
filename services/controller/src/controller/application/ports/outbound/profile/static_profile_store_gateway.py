from abc import ABC, abstractmethod

from controller.domain.profile.model_variant_static_profile import (
    ModelVariantStaticProfile,
)


class StaticProfileStoreGateway(ABC):
    @abstractmethod
    async def get_all_model_variant_static_profiles(
        self,
    ) -> list[ModelVariantStaticProfile]: ...
