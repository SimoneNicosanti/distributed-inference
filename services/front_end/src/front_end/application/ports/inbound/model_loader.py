from abc import ABC, abstractmethod

from artifacts.contracts.artifact_bundle import ArtifactBundle
from integration.model_manager.dto.model_dto import ModelDto
from integration.model_manager.dto.model_variant_dto import ModelVariantDto
from shared.artifact.artifact_ref import ArtifactRef


class ModelLoader(ABC):
    @abstractmethod
    async def register_model(self, model: ModelDto) -> None: ...

    @abstractmethod
    async def load_model_variant(
        self,
        model_variant: ModelVariantDto,
        artifact_bundle: ArtifactBundle,
    ) -> ArtifactRef: ...
