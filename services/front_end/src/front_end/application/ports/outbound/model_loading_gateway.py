from abc import ABC, abstractmethod

from artifacts.contracts.artifact_bundle import ArtifactBundle
from integration.model_manager.dto.model_dto import ModelDto
from integration.model_manager.dto.model_variant_dto import ModelVariantDto
from shared.artifact.artifact_ref import ArtifactRef
from shared.model.model_variant import ModelVariantId


class ModelManagerGateway(ABC):
    @abstractmethod
    async def register_model(self, model: ModelDto) -> None: ...

    @abstractmethod
    async def start_model_variant_registration(
        self,
        model_variant: ModelVariantDto,
    ) -> ArtifactRef: ...

    @abstractmethod
    async def finalize_model_variant_registration(
        self,
        model_variant_id: ModelVariantId,
    ) -> None: ...


class ModelArtifactUploader(ABC):
    @abstractmethod
    async def upload_model_artifact(
        self,
        artifact_ref: ArtifactRef,
        artifact_bundle: ArtifactBundle,
    ) -> None: ...
