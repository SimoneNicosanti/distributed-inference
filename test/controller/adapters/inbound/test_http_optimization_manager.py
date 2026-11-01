from typing import override
from uuid import UUID

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from controller.adapters.inbound.http_optimization_manager import (
    build_optimization_manager_router,
)
from controller.application.ports.inbound.optimization.optimization_manager import (
    OptimizationManager,
)
from controller.domain.optimization.request.flow_endpoint_placement import (
    AnyPlacement,
    FixedPlacement,
    FlowEndpointPlacement,
)
from controller.domain.optimization.request.local_worker_config import (
    NoLocalWorkerConfig,
    ValidLocalWorkerConfig,
)
from controller.domain.optimization.request.optimization_request import (
    OptimizationRequest,
    OptimizationRequestsByFlow,
)
from controller.domain.optimization.request.qos_constraints import QoSRequirements
from controller.domain.optimization.request.stream_info import StreamInfo
from integration.controller.http.optimization_http_paths import (
    OPTIMIZE_INFERENCE_DISTRIBUTION_PATH,
)
from integration.controller.http.optimization_http_schema import (
    OptimizeInferenceDistributionRequest,
)
from integration.dto.optimization.optimization_dto import (
    AnyPlacementDto,
    FixedPlacementDto,
    FlowEndpointPlacementDto,
    NoLocalWorkerConfigDto,
    OptimizationRequestDto,
    QoSRequirementsDto,
    StreamInfoDto,
    ValidLocalWorkerConfigDto,
)
from shared.flow.flow import FlowId
from shared.model.model import ModelTask
from shared.model.model_variant import PerformanceMetric, PerformanceMetricType
from shared.service.service import ServiceId
from shared.user.user import UserId


class RecordingOptimizationManager(OptimizationManager):
    def __init__(self) -> None:
        self.calls: list[OptimizationRequestsByFlow] = []

    @override
    async def optimize_inference_distribution(
        self,
        requests_by_flow: OptimizationRequestsByFlow,
    ) -> None:
        self.calls.append(requests_by_flow)


def _request_dto(flow_id: FlowId) -> OptimizationRequestDto:
    return OptimizationRequestDto(
        flow_id=flow_id,
        task=ModelTask.CLASSIFICATION,
        stream_info=StreamInfoDto(arrival_rate=2.5),
        qos_requirements=QoSRequirementsDto(
            max_mean_response_time=0.5,
            min_model_performance=PerformanceMetric(
                type=PerformanceMetricType.MAP_AT_50,
                dataset="validation",
                value=0.8,
            ),
            max_energy_consumption=10,
        ),
        flow_endpoint_placement=FlowEndpointPlacementDto(
            ingress_placement=FixedPlacementDto(worker_id=ServiceId(id="ingress")),
            sink_placement=FixedPlacementDto(worker_id=ServiceId(id="sink")),
        ),
        local_worker_config=ValidLocalWorkerConfigDto(
            worker_id=ServiceId(id="ingress"),
            is_inference_runner=True,
        ),
    )


def _client(manager: OptimizationManager) -> TestClient:
    app = FastAPI()
    app.include_router(build_optimization_manager_router(manager))
    return TestClient(app)


def test_maps_request_to_domain_and_delegates_to_manager() -> None:
    manager = RecordingOptimizationManager()
    client = _client(manager)
    flow_id = FlowId(user_id=UserId())
    request_dto = _request_dto(flow_id)
    request = OptimizeInferenceDistributionRequest(requests=(request_dto,))

    response = client.post(
        OPTIMIZE_INFERENCE_DISTRIBUTION_PATH,
        json=request.model_dump(mode="json"),
    )

    assert response.status_code == 204
    assert manager.calls == [
        {
            flow_id: OptimizationRequest(
                flow_id=flow_id,
                task=request_dto.task,
                stream_info=StreamInfo(
                    arrival_rate=request_dto.stream_info.arrival_rate
                ),
                qos_requirements=QoSRequirements(
                    max_mean_response_time=(
                        request_dto.qos_requirements.max_mean_response_time
                    ),
                    min_model_performance=(
                        request_dto.qos_requirements.min_model_performance
                    ),
                    max_energy_consumption=(
                        request_dto.qos_requirements.max_energy_consumption
                    ),
                ),
                flow_endpoint_placement=FlowEndpointPlacement(
                    ingress_placement=FixedPlacement(worker_id=ServiceId(id="ingress")),
                    sink_placement=FixedPlacement(worker_id=ServiceId(id="sink")),
                ),
                local_worker_config=ValidLocalWorkerConfig(
                    worker_id=ServiceId(id="ingress"),
                    is_inference_runner=True,
                ),
            )
        }
    ]


@pytest.mark.parametrize("fixed_ingress", [False, True])
@pytest.mark.parametrize("fixed_sink", [False, True])
@pytest.mark.parametrize("is_inference_runner", [None, False, True])
def test_maps_all_placement_and_local_worker_variants(
    fixed_ingress: bool,
    fixed_sink: bool,
    is_inference_runner: bool | None,
) -> None:
    manager = RecordingOptimizationManager()
    flow_id = FlowId(user_id=UserId())
    ingress_id = ServiceId(id="ingress")
    sink_id = ServiceId(id="sink")
    dto = _request_dto(flow_id).model_copy(
        update={
            "flow_endpoint_placement": FlowEndpointPlacementDto(
                ingress_placement=(
                    FixedPlacementDto(worker_id=ingress_id)
                    if fixed_ingress
                    else AnyPlacementDto()
                ),
                sink_placement=(
                    FixedPlacementDto(worker_id=sink_id)
                    if fixed_sink
                    else AnyPlacementDto()
                ),
            ),
            "local_worker_config": (
                NoLocalWorkerConfigDto()
                if is_inference_runner is None
                else ValidLocalWorkerConfigDto(
                    worker_id=ingress_id,
                    is_inference_runner=is_inference_runner,
                )
            ),
        }
    )
    request = OptimizeInferenceDistributionRequest(requests=(dto,))
    payload = request.model_dump(mode="json")

    decoded = OptimizeInferenceDistributionRequest.model_validate_json(
        request.model_dump_json()
    )
    assert decoded.model_dump(mode="json") == payload
    assert OptimizeInferenceDistributionRequest.model_validate(payload) == decoded

    response = _client(manager).post(
        OPTIMIZE_INFERENCE_DISTRIBUTION_PATH,
        json=payload,
    )

    assert response.status_code == 204
    assert len(manager.calls) == 1
    actual = manager.calls[0][flow_id]
    assert actual.flow_endpoint_placement == FlowEndpointPlacement(
        ingress_placement=(
            FixedPlacement(worker_id=ingress_id) if fixed_ingress else AnyPlacement()
        ),
        sink_placement=(
            FixedPlacement(worker_id=sink_id) if fixed_sink else AnyPlacement()
        ),
    )
    assert actual.local_worker_config == (
        NoLocalWorkerConfig()
        if is_inference_runner is None
        else ValidLocalWorkerConfig(
            worker_id=ingress_id,
            is_inference_runner=is_inference_runner,
        )
    )


def test_preserves_independent_configurations_in_a_multi_flow_batch() -> None:
    manager = RecordingOptimizationManager()
    local_worker_id = ServiceId(id="shared-local-worker")
    first = _request_dto(FlowId(user_id=UserId())).model_copy(
        update={
            "local_worker_config": ValidLocalWorkerConfigDto(
                worker_id=local_worker_id,
                is_inference_runner=False,
            ),
        }
    )
    second = _request_dto(FlowId(user_id=UserId())).model_copy(
        update={
            "flow_endpoint_placement": FlowEndpointPlacementDto(
                ingress_placement=AnyPlacementDto(),
                sink_placement=AnyPlacementDto(),
            ),
            "local_worker_config": ValidLocalWorkerConfigDto(
                worker_id=local_worker_id,
                is_inference_runner=True,
            ),
        }
    )
    request = OptimizeInferenceDistributionRequest(requests=(first, second))

    response = _client(manager).post(
        OPTIMIZE_INFERENCE_DISTRIBUTION_PATH,
        json=request.model_dump(mode="json"),
    )

    assert response.status_code == 204
    assert len(manager.calls) == 1
    actual = manager.calls[0]
    assert set(actual) == {first.flow_id, second.flow_id}
    assert actual[first.flow_id].local_worker_config == ValidLocalWorkerConfig(
        worker_id=local_worker_id,
        is_inference_runner=False,
    )
    assert actual[second.flow_id].local_worker_config == ValidLocalWorkerConfig(
        worker_id=local_worker_id,
        is_inference_runner=True,
    )
    assert actual[first.flow_id].flow_endpoint_placement == FlowEndpointPlacement(
        ingress_placement=FixedPlacement(worker_id=ServiceId(id="ingress")),
        sink_placement=FixedPlacement(worker_id=ServiceId(id="sink")),
    )
    assert actual[second.flow_id].flow_endpoint_placement == FlowEndpointPlacement(
        ingress_placement=AnyPlacement(),
        sink_placement=AnyPlacement(),
    )


@pytest.mark.parametrize(
    "worker_id",
    [ServiceId(id="local-worker"), ServiceId(id=UUID(int=123))],
    ids=["named", "uuid"],
)
def test_transmits_named_and_uuid_worker_ids(worker_id: ServiceId) -> None:
    manager = RecordingOptimizationManager()
    flow_id = FlowId(user_id=UserId())
    dto = _request_dto(flow_id).model_copy(
        update={
            "flow_endpoint_placement": FlowEndpointPlacementDto(
                ingress_placement=FixedPlacementDto(worker_id=worker_id),
                sink_placement=FixedPlacementDto(worker_id=worker_id),
            ),
            "local_worker_config": ValidLocalWorkerConfigDto(
                worker_id=worker_id,
                is_inference_runner=False,
            ),
        }
    )
    request = OptimizeInferenceDistributionRequest(requests=(dto,))
    payload = request.model_dump(mode="json")
    decoded = OptimizeInferenceDistributionRequest.model_validate_json(
        request.model_dump_json()
    )
    assert decoded.model_dump(mode="json") == payload

    response = _client(manager).post(
        OPTIMIZE_INFERENCE_DISTRIBUTION_PATH,
        json=payload,
    )

    assert response.status_code == 204
    actual = manager.calls[0][flow_id]
    ingress = actual.flow_endpoint_placement.ingress_placement
    sink = actual.flow_endpoint_placement.sink_placement
    local = actual.local_worker_config
    assert isinstance(ingress, FixedPlacement)
    assert isinstance(sink, FixedPlacement)
    assert isinstance(local, ValidLocalWorkerConfig)
    assert str(ingress.worker_id.id) == str(worker_id.id)
    assert str(sink.worker_id.id) == str(worker_id.id)
    assert str(local.worker_id.id) == str(worker_id.id)
    assert local.is_inference_runner is False


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("flow_endpoint_placement", None),
        ("flow_endpoint_placement", {}),
        (
            "flow_endpoint_placement",
            {"ingress_placement": {"kind": "any"}},
        ),
        (
            "flow_endpoint_placement",
            {"sink_placement": {"kind": "any"}},
        ),
        ("local_worker_config", None),
        ("local_worker_config", {}),
        ("local_worker_config", {"kind": "unknown"}),
        ("local_worker_config", {"kind": "valid", "is_inference_runner": False}),
        ("local_worker_config", {"kind": "valid", "worker_id": {"id": "local"}}),
        (
            "local_worker_config",
            {"kind": "valid", "worker_id": {}, "is_inference_runner": False},
        ),
        (
            "local_worker_config",
            {"worker_id": {"id": "local"}, "is_inference_runner": False},
        ),
        (
            "local_worker_config",
            {
                "kind": "valid",
                "worker_id": {"id": "local"},
                "is_inference_runner": None,
            },
        ),
        (
            "local_worker_config",
            {
                "kind": "valid",
                "worker_id": {"id": "local"},
                "is_inference_runner": "invalid",
            },
        ),
        ("local_worker_config", {"kind": "none", "worker_id": {"id": "local"}}),
        ("local_worker_config", {"kind": "none", "is_inference_runner": False}),
    ],
)
def test_rejects_invalid_request_configurations(
    field: str, invalid_value: object
) -> None:
    manager = RecordingOptimizationManager()
    payload = _request_dto(FlowId(user_id=UserId())).model_dump(mode="json")
    payload[field] = invalid_value

    response = _client(manager).post(
        OPTIMIZE_INFERENCE_DISTRIBUTION_PATH,
        json={"requests": [payload]},
    )

    assert response.status_code == 422
    assert manager.calls == []


@pytest.mark.parametrize("endpoint", ["ingress_placement", "sink_placement"])
@pytest.mark.parametrize(
    "invalid_placement",
    [
        None,
        {},
        {"kind": "unknown"},
        {"worker_id": {"id": "worker"}},
        {"kind": "fixed"},
        {"kind": "fixed", "worker_id": {}},
        {"kind": "fixed", "worker_id": {"id": None}},
        {"kind": "fixed", "worker_id": {"id": []}},
        {"kind": "fixed", "worker_id": "worker"},
        {"kind": "any", "worker_id": {"id": "worker"}},
    ],
)
def test_rejects_invalid_endpoint_placements(
    endpoint: str, invalid_placement: object
) -> None:
    manager = RecordingOptimizationManager()
    payload = _request_dto(FlowId(user_id=UserId())).model_dump(mode="json")
    payload["flow_endpoint_placement"][endpoint] = invalid_placement

    response = _client(manager).post(
        OPTIMIZE_INFERENCE_DISTRIBUTION_PATH,
        json={"requests": [payload]},
    )

    assert response.status_code == 422
    assert manager.calls == []


@pytest.mark.parametrize("field", ["flow_endpoint_placement", "local_worker_config"])
def test_requires_explicit_configuration_fields(field: str) -> None:
    manager = RecordingOptimizationManager()
    payload = _request_dto(FlowId(user_id=UserId())).model_dump(mode="json")
    del payload[field]

    response = _client(manager).post(
        OPTIMIZE_INFERENCE_DISTRIBUTION_PATH,
        json={"requests": [payload]},
    )

    assert response.status_code == 422
    assert manager.calls == []


def test_rejects_legacy_ingress_only_requests_without_inventing_configuration() -> None:
    manager = RecordingOptimizationManager()
    payload = _request_dto(FlowId(user_id=UserId())).model_dump(mode="json")
    del payload["flow_endpoint_placement"]
    del payload["local_worker_config"]
    payload["ingress_id"] = {"id": "ingress"}

    response = _client(manager).post(
        OPTIMIZE_INFERENCE_DISTRIBUTION_PATH,
        json={"requests": [payload]},
    )

    assert response.status_code == 422
    assert manager.calls == []


def test_rejects_empty_request_batch() -> None:
    manager = RecordingOptimizationManager()

    response = _client(manager).post(
        OPTIMIZE_INFERENCE_DISTRIBUTION_PATH,
        json={"requests": []},
    )

    assert response.status_code == 422
    assert manager.calls == []


def test_rejects_duplicate_flow_ids() -> None:
    manager = RecordingOptimizationManager()
    request = _request_dto(FlowId(user_id=UserId())).model_dump(mode="json")

    response = _client(manager).post(
        OPTIMIZE_INFERENCE_DISTRIBUTION_PATH,
        json={"requests": [request, request]},
    )

    assert response.status_code == 422
    assert manager.calls == []
