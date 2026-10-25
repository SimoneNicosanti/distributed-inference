import os
import time
from pathlib import Path
from uuid import uuid4

import httpx
import onnx
import pytest
from redis import Redis
from test.system.test_upload_model import _build_model_archive, _read_model_inputs

from integration.controller.http.optimization_http_paths import (
    OPTIMIZE_INFERENCE_DISTRIBUTION_PATH,
)
from integration.controller.http.optimization_http_schema import (
    OptimizeInferenceDistributionRequest,
)
from integration.dto.model.model_dto import ModelDto
from integration.dto.model.model_variant_dto import (
    DynamicShapeValuesDto,
    ModelVariantDto,
    ModelVariantInfoDto,
)
from integration.dto.optimization.optimization_dto import (
    OptimizationRequestDto,
    QoSRequirementsDto,
    StreamInfoDto,
)
from integration.dto.worker_identity.worker_identity_dto import WorkerIdentityDto
from integration.dto.worker_profiles.execution_profile_dto import (
    ModelExecutionProfileDto,
)
from integration.front_end.http.front_end_http_paths import (
    FRONT_END_LOAD_MODEL_VARIANT_PATH,
    FRONT_END_LOCAL_WORKER_IDENTITY_PATH,
    FRONT_END_REGISTER_MODEL_PATH,
)
from integration.front_end.http.front_end_http_schema import LoadModelVariantResponse
from integration.model_manager.http.model_manager_http_schema import (
    RegisterModelRequest,
)
from integration.redis.worker_profiles.execution_profile_keys import (
    build_execution_profile_key_per_worker,
)
from integration.redis.worker_profiles.resource_profile_keys import (
    build_resource_profile_key_per_worker,
)
from shared.flow.flow import FlowId
from shared.model.model import (
    CNNArchitectureInfo,
    ModelId,
    ModelTask,
    ModelVisibility,
)
from shared.model.model_variant import (
    ModelVariantFormat,
    ModelVariantId,
    NumericPrecision,
    PerformanceMetric,
    PerformanceMetricType,
    QuantizationType,
)
from shared.service.service import ServiceId
from shared.user.user import UserId

ARRIVAL_RATES = (0.001, 0.002, 0.003, 0.004, 0.005)
MAX_RESPONSE_TIMES_S = (300.0,) * 5
MODEL_ACCURACIES = (0.75, 0.80, 0.85, 0.90)


def _wait_for_worker_profiles(
    redis: Redis,
    worker_id: ServiceId,
    model_variant_ids: set[ModelVariantId],
    timeout_s: float,
) -> None:
    execution_profile_key = build_execution_profile_key_per_worker(worker_id)
    resource_profile_key = build_resource_profile_key_per_worker(worker_id)
    deadline = time.monotonic() + timeout_s

    while True:
        values = redis.hvals(execution_profile_key)
        profiled_model_ids = {
            ModelExecutionProfileDto.model_validate_json(value).model_variant_id
            for value in values
        }
        if (
            redis.exists(resource_profile_key)
            and model_variant_ids <= profiled_model_ids
        ):
            return

        if time.monotonic() >= deadline:
            missing = model_variant_ids - profiled_model_ids
            pytest.fail(
                "Worker profiles were not ready before timeout; "
                f"missing model variants: {sorted(map(repr, missing))}"
            )
        time.sleep(1.0)


@pytest.mark.integration
def test_optimization_request(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    front_end_base_url = os.getenv("FRONT_END_BASE_URL")
    if front_end_base_url is None:
        pytest.skip("Set FRONT_END_BASE_URL to run system tests")

    controller_base_url = os.getenv("CONTROLLER_BASE_URL")
    if controller_base_url is None:
        pytest.skip("Set CONTROLLER_BASE_URL to run the optimization test")

    redis_url = os.getenv("REDIS_URL")
    if redis_url is None:
        pytest.skip("Set REDIS_URL to wait for worker profiling")

    model_path_value = os.getenv("MODEL_PATH")
    if model_path_value is None:
        pytest.skip("Set MODEL_PATH to an ONNX model to run the optimization test")

    model_path = Path(model_path_value)
    if not model_path.is_file():
        pytest.fail(f"MODEL_PATH is not a file: {model_path}")

    onnx_model = onnx.load(model_path, load_external_data=False)
    onnx.checker.check_model(onnx_model)
    inputs_info, dynamic_dimension_values = _read_model_inputs(onnx_model)
    fixed_dynamic_dimension_values = {
        name: DynamicShapeValuesDto(values=frozenset({1}))
        for name in dynamic_dimension_values
    }

    archive_path = tmp_path / "artifact.zip"
    _build_model_archive(model_path, archive_path)

    user_id = UserId()
    metric_dataset = f"system-test-{uuid4().hex}"
    variants: list[ModelVariantDto] = []

    with httpx.Client(base_url=front_end_base_url, timeout=300.0) as client:
        identity_response = client.get(FRONT_END_LOCAL_WORKER_IDENTITY_PATH)
        assert identity_response.status_code == 200, identity_response.text
        identity = WorkerIdentityDto.model_validate(identity_response.json())
        ingress_id = ServiceId(id=identity.service_id)

        for index, accuracy in enumerate(MODEL_ACCURACIES, start=1):
            model_id = ModelId(
                owner_id=user_id,
                model_name=f"optimization-system-test-{uuid4().hex}",
            )
            model = ModelDto(
                model_id=model_id,
                visibility=ModelVisibility.PRIVATE,
                model_task=ModelTask.CLASSIFICATION,
                architecture_info=CNNArchitectureInfo(),
            )
            variant = ModelVariantDto(
                id=ModelVariantId(model_id=model_id, variant_tag=f"model-{index}"),
                info=ModelVariantInfoDto(
                    precision=NumericPrecision.FP32,
                    quantization=QuantizationType.NONE,
                    accuracies=[
                        PerformanceMetric(
                            type=PerformanceMetricType.MAP_AT_50,
                            dataset=metric_dataset,
                            value=accuracy,
                        )
                    ],
                    format=ModelVariantFormat.ONNX,
                    inputs_info=inputs_info,
                    dynamic_dimension_values=fixed_dynamic_dimension_values,
                ),
            )

            register_response = client.post(
                FRONT_END_REGISTER_MODEL_PATH,
                json=RegisterModelRequest(model_dto=model).model_dump(mode="json"),
            )
            assert register_response.status_code == 200, register_response.text

            with archive_path.open("rb") as archive:
                upload_response = client.post(
                    FRONT_END_LOAD_MODEL_VARIANT_PATH,
                    data={"model_variant_json": variant.model_dump_json()},
                    files={
                        "artifact": (
                            archive_path.name,
                            archive,
                            "application/zip",
                        )
                    },
                )
            assert upload_response.status_code == 200, upload_response.text
            upload = LoadModelVariantResponse.model_validate(upload_response.json())
            assert upload.artifact_ref.value
            variants.append(variant)

    profile_timeout_s = float(os.getenv("WORKER_PROFILE_TIMEOUT_S", "600"))
    with Redis.from_url(redis_url, decode_responses=True) as redis:
        _wait_for_worker_profiles(
            redis,
            ingress_id,
            {variant.id for variant in variants},
            profile_timeout_s,
        )

    requests = tuple(
        OptimizationRequestDto(
            flow_id=FlowId(user_id=user_id),
            task=ModelTask.CLASSIFICATION,
            stream_info=StreamInfoDto(arrival_rate=arrival_rate),
            qos_requirements=QoSRequirementsDto(
                max_mean_response_time=max_response_time,
                min_model_performance=PerformanceMetric(
                    type=PerformanceMetricType.MAP_AT_50,
                    dataset=metric_dataset,
                    value=0.70,
                ),
                max_energy_consumption=1_000_000.0,
            ),
            ingress_id=ingress_id,
        )
        for arrival_rate, max_response_time in zip(
            ARRIVAL_RATES,
            MAX_RESPONSE_TIMES_S,
            strict=True,
        )
    )
    request = OptimizeInferenceDistributionRequest(requests=requests)

    request_started_at = time.perf_counter()
    try:
        response = httpx.post(
            f"{controller_base_url.rstrip('/')}{OPTIMIZE_INFERENCE_DISTRIBUTION_PATH}",
            json=request.model_dump(mode="json"),
            timeout=300.0,
        )
    finally:
        request_elapsed_s = time.perf_counter() - request_started_at
        with capsys.disabled():
            print(
                f"Optimization request elapsed time: {request_elapsed_s:.3f} s",
                flush=True,
            )

    assert response.status_code == 204, response.text
