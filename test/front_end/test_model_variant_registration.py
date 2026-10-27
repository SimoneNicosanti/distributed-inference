from typing import cast
from unittest.mock import AsyncMock

import httpx
import pytest

from artifacts.contracts.artifact_bundle import ArtifactBundle
from front_end.adapters.outbound.http_model_manager_gateway import (
    HttpModelManagerGateway,
)
from front_end.application.loading.default_model_loader import DefaultModelLoader
from front_end.application.ports.outbound.model_loading_gateway import (
    ModelArtifactUploader,
    ModelManagerGateway,
)
from integration.dto.model.model_variant_dto import ModelVariantInfoDto
from integration.model_manager.http.model_manager_http_paths import (
    START_MODEL_VARIANT_REGISTRATION_PATH,
)
from integration.model_manager.http.model_manager_http_schema import (
    StartModelVariantRegistrationRequest,
    StartModelVariantRegistrationResponse,
)
from shared.artifact.artifact_ref import ArtifactRef
from shared.model.model import ModelId
from shared.model.model_variant import (
    ModelVariantFormat,
    ModelVariantId,
    NumericPrecision,
    QuantizationType,
)

_MODEL_ID = ModelId(model_name="model")
_MODEL_VARIANT_ID = ModelVariantId(model_id=_MODEL_ID, variant_tag="generated")
_MODEL_VARIANT_INFO = ModelVariantInfoDto(
    precision=NumericPrecision.FP32,
    quantization=QuantizationType.NONE,
    accuracies=[],
    format=ModelVariantFormat.ONNX,
    inputs_info={},
    dynamic_dimension_values={},
)
_ARTIFACT_REF = ArtifactRef(value="artifact")


@pytest.mark.asyncio
async def test_loader_finalizes_generated_model_variant_id() -> None:
    gateway = AsyncMock(spec=ModelManagerGateway)
    gateway.start_model_variant_registration.return_value = (
        _MODEL_VARIANT_ID,
        _ARTIFACT_REF,
    )
    uploader = AsyncMock(spec=ModelArtifactUploader)
    artifact_bundle = cast(ArtifactBundle, object())

    result = await DefaultModelLoader(gateway, uploader).load_model_variant(
        _MODEL_ID,
        _MODEL_VARIANT_INFO,
        artifact_bundle,
    )

    assert result == (_MODEL_VARIANT_ID, _ARTIFACT_REF)
    gateway.finalize_model_variant_registration.assert_awaited_once_with(
        _MODEL_VARIANT_ID
    )
    uploader.upload_model_artifact.assert_awaited_once_with(
        artifact_ref=_ARTIFACT_REF,
        artifact_bundle=artifact_bundle,
    )


@pytest.mark.asyncio
async def test_http_gateway_returns_generated_model_variant_id() -> None:
    async def handle_request(request: httpx.Request) -> httpx.Response:
        assert request.url.path == START_MODEL_VARIANT_REGISTRATION_PATH
        payload = StartModelVariantRegistrationRequest.model_validate_json(
            request.content
        )
        assert payload.model_id == _MODEL_ID
        assert payload.model_variant_info_dto == _MODEL_VARIANT_INFO
        response = StartModelVariantRegistrationResponse(
            model_variant_id=_MODEL_VARIANT_ID,
            artifact_ref=_ARTIFACT_REF,
        )
        return httpx.Response(200, json=response.model_dump(mode="json"))

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handle_request)
    ) as client:
        result = await HttpModelManagerGateway(
            client,
            "http://model-manager",
        ).start_model_variant_registration(_MODEL_ID, _MODEL_VARIANT_INFO)

    assert result == (_MODEL_VARIANT_ID, _ARTIFACT_REF)
