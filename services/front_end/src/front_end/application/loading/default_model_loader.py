from typing import override

from artifacts.contracts.artifact_bundle import ArtifactBundle
from front_end.application.ports.inbound.model_loader import ModelLoader
from front_end.application.ports.outbound.model_loading_gateway import (
    ModelArtifactUploader,
    ModelManagerGateway,
)
from integration.dto.model.model_dto import ModelDto
from integration.dto.model.model_variant_dto import ModelVariantInfoDto
from shared.artifact.artifact_ref import ArtifactRef
from shared.model.model import ModelId
from shared.model.model_variant import ModelVariantId


class DefaultModelLoader(ModelLoader):
    def __init__(
        self,
        model_manager_gateway: ModelManagerGateway,
        model_artifact_uploader: ModelArtifactUploader,
    ) -> None:
        self._model_manager_gateway = model_manager_gateway
        self._model_artifact_uploader = model_artifact_uploader

    @override
    async def register_model(self, model: ModelDto) -> None:
        await self._model_manager_gateway.register_model(model)

    @override
    async def load_model_variant(
        self,
        model_id: ModelId,
        model_variant_info: ModelVariantInfoDto,
        artifact_bundle: ArtifactBundle,
    ) -> tuple[ModelVariantId, ArtifactRef]:
        model_variant_id, artifact_ref = (
            await self._model_manager_gateway.start_model_variant_registration(
                model_id,
                model_variant_info,
            )
        )
        await self._model_artifact_uploader.upload_model_artifact(
            artifact_ref=artifact_ref,
            artifact_bundle=artifact_bundle,
        )
        await self._model_manager_gateway.finalize_model_variant_registration(
            model_variant_id
        )

        return model_variant_id, artifact_ref
