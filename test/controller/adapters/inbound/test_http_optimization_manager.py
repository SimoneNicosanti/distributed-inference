from typing import override

from fastapi import FastAPI
from fastapi.testclient import TestClient

from controller.adapters.inbound.http_optimization_manager import (
    build_optimization_manager_router,
)
from controller.application.ports.inbound.optimization.optimization_manager import (
    OptimizationManager,
)
from controller.domain.optimization.optimization_request import (
    OptimizationRequest,
    OptimizationRequestsByFlow,
)
from controller.domain.optimization.qos_constraints import QoSRequirements
from controller.domain.optimization.stream_info import StreamInfo
from integration.controller.http.optimization_http_paths import (
    OPTIMIZE_INFERENCE_DISTRIBUTION_PATH,
)
from integration.controller.http.optimization_http_schema import (
    OptimizeInferenceDistributionRequest,
)
from integration.dto.optimization.optimization_dto import (
    OptimizationRequestDto,
    QoSRequirementsDto,
    StreamInfoDto,
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
        ingress_id=ServiceId(),
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
                ingress_id=request_dto.ingress_id,
            )
        }
    ]


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
