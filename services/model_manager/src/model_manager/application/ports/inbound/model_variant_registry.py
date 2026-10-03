from abc import ABC, abstractmethod

from shared.artifact.artifact_ref import ArtifactRef
from model_manager.domain.model_variant import ModelVariant
from shared.model.model_variant import ModelVariantId


class ModelVariantRegistry(ABC):
    @abstractmethod
    async def start_model_variant_registration(
        self, model_variant: ModelVariant
    ) -> ArtifactRef: ...

    @abstractmethod
    async def finalize_model_variant_registration(
        self, model_variant_id: ModelVariantId
    ) -> None: ...
