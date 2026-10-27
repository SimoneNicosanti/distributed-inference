from abc import ABC, abstractmethod

from model_manager.domain.model_variant import ModelVariantInfo
from shared.artifact.artifact_ref import ArtifactRef
from shared.model.model import ModelId
from shared.model.model_variant import ModelVariantId


class ModelVariantRegistry(ABC):
    @abstractmethod
    async def start_model_variant_registration(
        self, model_id: ModelId, model_variant_info: ModelVariantInfo
    ) -> tuple[ModelVariantId, ArtifactRef]: ...

    @abstractmethod
    async def finalize_model_variant_registration(
        self, model_variant_id: ModelVariantId
    ) -> None: ...
