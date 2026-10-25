import os
from pathlib import Path
from uuid import uuid4

import httpx
import onnx
import pytest
from test.system.test_upload_model import _build_model_archive, _read_model_inputs

from integration.dto.model.model_dto import ModelDto
from integration.dto.model.model_variant_dto import (
    ModelVariantDto,
    ModelVariantInfoDto,
)
from integration.front_end.http.front_end_http_paths import (
    FRONT_END_LOAD_MODEL_VARIANT_PATH,
    FRONT_END_REGISTER_MODEL_PATH,
)
from integration.front_end.http.front_end_http_schema import LoadModelVariantResponse
from integration.model_manager.http.model_manager_http_paths import (
    GENERATE_MODEL_PARTITION_PATH,
)
from integration.model_manager.http.model_manager_http_schema import (
    GenerateModelPartitionRequest,
    GenerateModelPartitionResponse,
    RegisterModelRequest,
)
from shared.model.model import (
    CNNArchitectureInfo,
    ModelId,
    ModelTask,
    ModelVisibility,
)
from shared.model.model_partition import ModelPartitionId
from shared.model.model_variant import (
    ModelVariantFormat,
    ModelVariantId,
    NumericPrecision,
    PerformanceMetric,
    PerformanceMetricType,
    QuantizationType,
)
from shared.user.user import SYSTEM_USER_ID


def _split_topological_layers(
    model: onnx.ModelProto,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    layers = tuple(node.name for node in model.graph.node)
    if len(layers) < 3:
        raise ValueError("ONNX model must contain at least three named layers")
    if any(not layer for layer in layers):
        raise ValueError("Every ONNX node must have a name")
    if len(layers) != len(set(layers)):
        raise ValueError("ONNX node names must be unique")

    x = len(layers) // 3
    y = 2 * len(layers) // 3
    partitions = layers[:x], layers[x:y], layers[y:]
    if any(not partition for partition in partitions):
        raise ValueError("Every model partition must contain at least one layer")

    return partitions


@pytest.mark.integration
def test_partition_model(tmp_path: Path) -> None:
    front_end_base_url = os.getenv("FRONT_END_BASE_URL")
    if front_end_base_url is None:
        pytest.skip("Set FRONT_END_BASE_URL to run system tests")

    model_manager_base_url = os.getenv("MODEL_MANAGER_BASE_URL")
    if model_manager_base_url is None:
        pytest.skip("Set MODEL_MANAGER_BASE_URL to run the partition test")

    model_path_value = os.getenv("MODEL_PATH")
    if model_path_value is None:
        pytest.skip("Set MODEL_PATH to an ONNX model to run the partition test")

    model_path = Path(model_path_value)
    if not model_path.is_file():
        pytest.fail(f"MODEL_PATH is not a file: {model_path}")

    onnx_model = onnx.load(model_path, load_external_data=False)
    onnx.checker.check_model(onnx_model)
    partitions = _split_topological_layers(onnx_model)
    inputs_info, dynamic_dimension_values = _read_model_inputs(onnx_model)

    archive_path = tmp_path / "artifact.zip"
    _build_model_archive(model_path, archive_path)

    unique_id = uuid4()
    model_id = ModelId(
        owner_id=SYSTEM_USER_ID,
        model_name=f"partition-system-test-{unique_id.hex}",
    )
    model = ModelDto(
        model_id=model_id,
        visibility=ModelVisibility.PRIVATE,
        model_task=ModelTask.CLASSIFICATION,
        architecture_info=CNNArchitectureInfo(),
    )
    variant = ModelVariantDto(
        id=ModelVariantId(model_id=model_id, variant_tag="fp32"),
        info=ModelVariantInfoDto(
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

    with httpx.Client(base_url=front_end_base_url, timeout=300.0) as client:
        register_response = client.post(
            FRONT_END_REGISTER_MODEL_PATH,
            json=RegisterModelRequest(model_dto=model).model_dump(mode="json"),
        )
        assert register_response.status_code == 200, register_response.text

        with archive_path.open("rb") as archive:
            upload_response = client.post(
                FRONT_END_LOAD_MODEL_VARIANT_PATH,
                data={"model_variant_json": variant.model_dump_json()},
                files={"artifact": (archive_path.name, archive, "application/zip")},
            )

        assert upload_response.status_code == 200, upload_response.text
        upload = LoadModelVariantResponse.model_validate(upload_response.json())
        assert upload.artifact_ref.value

    artifact_refs: set[str] = set()
    with httpx.Client(base_url=model_manager_base_url, timeout=300.0) as client:
        for layers in partitions:
            request = GenerateModelPartitionRequest(
                model_variant_id=variant.id,
                layers=list(layers),
            )
            response = client.post(
                GENERATE_MODEL_PARTITION_PATH,
                json=request.model_dump(mode="json"),
            )
            assert response.status_code == 200, response.text

            partition = GenerateModelPartitionResponse.model_validate(response.json())
            assert partition.model_partition_id == ModelPartitionId(
                model_variant_id=variant.id,
                layers=layers,
            )
            assert partition.artifact_ref.value
            artifact_refs.add(partition.artifact_ref.value)

    assert len(artifact_refs) == 3
