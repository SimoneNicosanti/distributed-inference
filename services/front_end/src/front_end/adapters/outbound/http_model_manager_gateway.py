from typing import override

import httpx

from front_end.application.ports.outbound.model_loading_gateway import (
    ModelManagerGateway,
)
from integration.dto.model.model_dto import ModelDto
from integration.dto.model.model_variant_dto import ModelVariantInfoDto
from integration.model_manager.http.model_manager_http_paths import (
    FINALIZE_MODEL_VARIANT_REGISTRATION_PATH,
    REGISTER_MODEL_PATH,
    START_MODEL_VARIANT_REGISTRATION_PATH,
)
from integration.model_manager.http.model_manager_http_schema import (
    FinalizeModelVariantRegistrationRequest,
    RegisterModelRequest,
    StartModelVariantRegistrationRequest,
    StartModelVariantRegistrationResponse,
)
from shared.artifact.artifact_ref import ArtifactRef
from shared.model.model import ModelId
from shared.model.model_variant import ModelVariantId


class HttpModelManagerGateway(ModelManagerGateway):
    def __init__(
        self,
        http_client: httpx.AsyncClient,
        base_url: str,
    ) -> None:
        self._http_client = http_client
        self._base_url = base_url.rstrip("/")

    @override
    async def register_model(self, model: ModelDto) -> None:
        request = RegisterModelRequest(model_dto=model)
        response = await self._http_client.post(
            self._build_url(REGISTER_MODEL_PATH),
            json=request.model_dump(mode="json"),
        )
        response.raise_for_status()

    @override
    async def start_model_variant_registration(
        self,
        model_id: ModelId,
        model_variant_info: ModelVariantInfoDto,
    ) -> tuple[ModelVariantId, ArtifactRef]:
        request = StartModelVariantRegistrationRequest(
            model_id=model_id,
            model_variant_info_dto=model_variant_info,
        )
        response = await self._http_client.post(
            self._build_url(START_MODEL_VARIANT_REGISTRATION_PATH),
            json=request.model_dump(mode="json"),
        )
        response.raise_for_status()

        response_dto = StartModelVariantRegistrationResponse.model_validate(
            response.json()
        )
        return response_dto.model_variant_id, response_dto.artifact_ref

    @override
    async def finalize_model_variant_registration(
        self,
        model_variant_id: ModelVariantId,
    ) -> None:
        request = FinalizeModelVariantRegistrationRequest(
            model_variant_id=model_variant_id
        )
        response = await self._http_client.post(
            self._build_url(FINALIZE_MODEL_VARIANT_REGISTRATION_PATH),
            json=request.model_dump(mode="json"),
        )
        response.raise_for_status()

    def _build_url(self, path: str) -> str:
        return f"{self._base_url}{path}"
