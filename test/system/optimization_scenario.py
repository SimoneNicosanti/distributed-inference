import os
import time
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import httpx
import onnx
import pytest
from redis import Redis
from test.system.model_support import _build_model_archive, _read_model_inputs
from test.system.optimization_readiness import (
    wait_for_flow_plans,
    wait_for_http,
    wait_for_worker_profiles,
)

from integration.controller.http.optimization_http_paths import (
    OPTIMIZE_INFERENCE_DISTRIBUTION_PATH,
)
from integration.controller.http.optimization_http_schema import (
    OptimizeInferenceDistributionRequest,
)
from integration.dto.flow_plan.flow_plan_dto import FlowPlanDto
from integration.dto.model.model_dto import ModelDto
from integration.dto.model.model_variant_dto import (
    DynamicShapeValuesDto,
    ModelVariantInfoDto,
)
from integration.dto.optimization.optimization_dto import (
    FixedPlacementDto,
    FlowEndpointPlacementDto,
    OptimizationRequestDto,
    QoSRequirementsDto,
    StreamInfoDto,
    ValidLocalWorkerConfigDto,
)
from integration.dto.worker_identity.worker_identity_dto import WorkerIdentityDto
from integration.front_end.http.front_end_http_paths import (
    FRONT_END_LOAD_MODEL_VARIANT_PATH,
    FRONT_END_LOCAL_WORKER_IDENTITY_PATH,
    FRONT_END_REGISTER_MODEL_PATH,
)
from integration.front_end.http.front_end_http_schema import LoadModelVariantResponse
from integration.model_manager.http.model_manager_http_schema import (
    RegisterModelRequest,
    StartModelVariantRegistrationRequest,
)
from shared.flow.flow import FlowId
from shared.model.model import (
    CNNArchitectureInfo,
    ModelId,
    ModelTask,
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

ARRIVAL_RATES = (0.01, 0.02, 0.03, 0.04, 0.05)
MAX_RESPONSE_TIMES_S = (100.0, 150.0, 200.0, 250.0, 300.0)
MODEL_ACCURACIES = (0.75, 0.80, 0.85, 0.90)


@dataclass(frozen=True)
class OptimizationScenario:
    front_end_base_url: str
    model_path: Path
    flow_plans: dict[FlowId, FlowPlanDto]
    model_variant_ids: frozenset[ModelVariantId]
    endpoint_worker_id: ServiceId


def prepare_optimization_scenario(tmp_path: Path) -> OptimizationScenario:
    front_end_base_url = os.getenv("FRONT_END_BASE_URL")
    if front_end_base_url is None:
        pytest.skip("Set FRONT_END_BASE_URL to run system tests")

    controller_base_url = os.getenv("CONTROLLER_BASE_URL")
    if controller_base_url is None:
        pytest.fail("Set CONTROLLER_BASE_URL when FRONT_END_BASE_URL is configured")

    redis_url = os.getenv("REDIS_URL")
    if redis_url is None:
        pytest.fail("Set REDIS_URL when FRONT_END_BASE_URL is configured")

    model_path_value = os.getenv("MODEL_PATH")
    if model_path_value is None:
        pytest.fail("Set MODEL_PATH when FRONT_END_BASE_URL is configured")

    model_path = Path(model_path_value)
    if not model_path.is_file():
        pytest.fail(f"MODEL_PATH is not a file: {model_path}")

    user_id = UserId(id=uuid4())
    flow_ids = tuple(FlowId(user_id=user_id, flow_id=uuid4()) for _ in ARRIVAL_RATES)

    onnx_model = onnx.load(model_path, load_external_data=False)
    onnx.checker.check_model(onnx_model)
    inputs_info, dynamic_dimension_values = _read_model_inputs(onnx_model)
    fixed_dynamic_dimension_values = {
        name: DynamicShapeValuesDto(values=frozenset({1}))
        for name in dynamic_dimension_values
    }

    archive_path = tmp_path / "artifact.zip"
    _build_model_archive(model_path, archive_path)

    metric_dataset = f"system-test-{uuid4().hex}"
    model_variant_ids: list[ModelVariantId] = []

    wait_for_http(front_end_base_url, FRONT_END_LOCAL_WORKER_IDENTITY_PATH)
    with httpx.Client(base_url=front_end_base_url, timeout=300.0) as client:
        identity_response = client.get(FRONT_END_LOCAL_WORKER_IDENTITY_PATH)
        assert identity_response.status_code == 200, identity_response.text
        identity = WorkerIdentityDto.model_validate(identity_response.json())
        ingress_id = ServiceId(id=identity.service_id)

        for accuracy in MODEL_ACCURACIES:
            model_id = ModelId(
                model_name=f"optimization-system-test-{uuid4().hex}",
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
                    data={"model_variant_json": registration.model_dump_json()},
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
            model_variant_ids.append(upload.model_variant_id)

    profile_timeout_s = float(os.getenv("WORKER_PROFILE_TIMEOUT_S", "600"))
    wait_for_worker_profiles(
        redis_url, ingress_id, set(model_variant_ids), profile_timeout_s
    )

    requests = tuple(
        OptimizationRequestDto(
            flow_id=flow_id,
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
            flow_endpoint_placement=FlowEndpointPlacementDto(
                ingress_placement=FixedPlacementDto(worker_id=ingress_id),
                sink_placement=FixedPlacementDto(worker_id=ingress_id),
            ),
            local_worker_config=ValidLocalWorkerConfigDto(
                worker_id=ingress_id,
                is_inference_runner=True,
            ),
        )
        for flow_id, arrival_rate, max_response_time in zip(
            flow_ids,
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
        print(
            f"Optimization request elapsed time: {request_elapsed_s:.3f} s",
            flush=True,
        )

    assert response.status_code == 204, response.text

    with Redis.from_url(redis_url, decode_responses=True) as redis:
        plans = wait_for_flow_plans(
            redis,
            flow_ids,
            float(os.getenv("PLAN_DEPLOYMENT_TIMEOUT_S", "120")),
            ingress_id,
        )
    return OptimizationScenario(
        front_end_base_url=front_end_base_url,
        model_path=model_path,
        flow_plans=plans,
        model_variant_ids=frozenset(model_variant_ids),
        endpoint_worker_id=ingress_id,
    )
