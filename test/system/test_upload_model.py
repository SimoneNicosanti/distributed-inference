import os
import zipfile
from pathlib import Path, PurePosixPath
from uuid import uuid4

import httpx
import onnx
import pytest

from artifacts.contracts.artifact_manifest import ArtifactFileInfo, ArtifactManifest
from integration.dto.model.model_dto import ModelDto
from integration.dto.model.model_variant_dto import (
    DynamicShapeValuesDto,
    InputInfoDto,
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


def _read_model_inputs(
    model: onnx.ModelProto,
) -> tuple[dict[str, InputInfoDto], dict[str, DynamicShapeValuesDto]]:
    initializer_names = {initializer.name for initializer in model.graph.initializer}
    inputs_info: dict[str, InputInfoDto] = {}
    dynamic_dimension_values: dict[str, DynamicShapeValuesDto] = {}

    for value_info in model.graph.input:
        if value_info.name in initializer_names:
            continue

        dimensions: list[str | int] = []
        for index, dimension in enumerate(value_info.type.tensor_type.shape.dim):
            if dimension.HasField("dim_value") and dimension.dim_value > 0:
                dimensions.append(dimension.dim_value)
                continue

            name = dimension.dim_param or f"{value_info.name}_dim_{index}"
            dimensions.append(name)
            dynamic_dimension_values.setdefault(
                name,
                DynamicShapeValuesDto(values=frozenset({1, 2, 4, 8})),
            )

        inputs_info[value_info.name] = InputInfoDto(shape=tuple(dimensions))

    if not inputs_info:
        raise ValueError("ONNX model has no runtime inputs")

    print(inputs_info)
    print(dynamic_dimension_values)

    return inputs_info, dynamic_dimension_values


def _build_model_archive(model_path: Path, archive_path: Path) -> None:
    artifact_model_path = PurePosixPath(model_path.name)
    manifest = ArtifactManifest(
        entrypoint_ppp=artifact_model_path,
        files_info=(ArtifactFileInfo(file_ppp=artifact_model_path),),
    )
    with zipfile.ZipFile(
        archive_path,
        mode="w",
        compression=zipfile.ZIP_STORED,
        allowZip64=True,
    ) as file:
        file.write(model_path, arcname=artifact_model_path.as_posix())
        file.writestr("manifest.json", manifest.model_dump_json())


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
