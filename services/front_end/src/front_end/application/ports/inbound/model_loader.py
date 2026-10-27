from abc import ABC, abstractmethod

from artifacts.contracts.artifact_bundle import ArtifactBundle
from integration.dto.model.model_dto import ModelDto
from integration.dto.model.model_variant_dto import ModelVariantInfoDto
from shared.artifact.artifact_ref import ArtifactRef
from shared.model.model import ModelId
from shared.model.model_variant import ModelVariantId


class ModelLoader(ABC):
    @abstractmethod
    async def register_model(self, model: ModelDto) -> None: ...

    @abstractmethod
    async def load_model_variant(
        self,
        model_id: ModelId,
        model_variant_info: ModelVariantInfoDto,
        artifact_bundle: ArtifactBundle,
    ) -> tuple[ModelVariantId, ArtifactRef]: ...
