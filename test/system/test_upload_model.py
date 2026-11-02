import os
from pathlib import Path
from uuid import uuid4

import httpx
import onnx
import pytest
from test.system.model_support import _build_model_archive, _read_model_inputs

from integration.dto.model.model_dto import ModelDto
from integration.dto.model.model_variant_dto import (
    ModelVariantInfoDto,
)
from integration.front_end.http.front_end_http_paths import (
    FRONT_END_LOAD_MODEL_VARIANT_PATH,
    FRONT_END_REGISTER_MODEL_PATH,
)
from integration.front_end.http.front_end_http_schema import LoadModelVariantResponse
from integration.model_manager.http.model_manager_http_schema import (
    RegisterModelRequest,
    StartModelVariantRegistrationRequest,
)
from shared.model.model import (
    CNNArchitectureInfo,
    ModelId,
    ModelTask,
)
from shared.model.model_variant import (
    ModelVariantFormat,
    NumericPrecision,
    PerformanceMetric,
    PerformanceMetricType,
    QuantizationType,
)


@pytest.mark.integration
def test_upload_model(tmp_path: Path) -> None:
    base_url = os.getenv("FRONT_END_BASE_URL")
    if base_url is None:
        pytest.skip("Set FRONT_END_BASE_URL to run system tests")

    model_path_value = os.getenv("MODEL_PATH")
    if model_path_value is None:
        pytest.skip("Set MODEL_PATH to an ONNX model to run the upload test")

    model_path = Path(model_path_value)
    if not model_path.is_file():
        pytest.fail(f"MODEL_PATH is not a file: {model_path}")

    onnx_model = onnx.load(model_path, load_external_data=False)
    onnx.checker.check_model(onnx_model)
    inputs_info, dynamic_dimension_values = _read_model_inputs(onnx_model)

    archive_path = tmp_path / "artifact.zip"
    _build_model_archive(model_path, archive_path)

    unique_id = uuid4()
    model_id = ModelId(
        model_name=f"system-test-{unique_id.hex}",
    )
    model = ModelDto(
        model_id=model_id,
        model_task=ModelTask.CLASSIFICATION,
        architecture_info=CNNArchitectureInfo(),
    )
    registration = StartModelVariantRegistrationRequest(
        model_id=model_id,
        model_variant_info_dto=ModelVariantInfoDto(
            precision=NumericPrecision.FP32,
            quantization=QuantizationType.NONE,
            accuracies=[
                PerformanceMetric(
                    type=PerformanceMetricType.MAP_AT_50,
                    dataset="system-test",
                    value=1.0,
                )
            ],
            format=ModelVariantFormat.ONNX,
            inputs_info=inputs_info,
            dynamic_dimension_values=dynamic_dimension_values,
        ),
    )

    with httpx.Client(base_url=base_url, timeout=120.0) as client:
        register_response = client.post(
            FRONT_END_REGISTER_MODEL_PATH,
            json=RegisterModelRequest(model_dto=model).model_dump(mode="json"),
        )
        assert register_response.status_code == 200, register_response.text

        with archive_path.open("rb") as archive:
            upload_response = client.post(
                FRONT_END_LOAD_MODEL_VARIANT_PATH,
                data={"model_variant_json": registration.model_dump_json()},
                files={"artifact": (archive_path.name, archive, "application/zip")},
            )

    assert upload_response.status_code == 200, upload_response.text
    response = LoadModelVariantResponse.model_validate(upload_response.json())
    assert response.model_variant_id.model_id == model_id
    assert response.artifact_ref.value
