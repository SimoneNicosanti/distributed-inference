from abc import ABC, abstractmethod

from artifacts.contracts.artifact_workspace import ArtifactWorkspace
from model_manager.domain.model import ModelInfo
from model_manager.domain.model_variant import (
    ModelVariant,
)
from model_manager.domain.model_variant_profile import ModelVariantProfile


class ModelProfiler(ABC):
    @abstractmethod
    async def profile_model_variant(
        self,
        artifact_workspace: ArtifactWorkspace,
        model_info: ModelInfo,
        model_variant: ModelVariant,
    ) -> ModelVariantProfile: ...
