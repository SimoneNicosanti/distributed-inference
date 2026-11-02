import httpx
import pytest

from front_end.adapters.outbound.optimization.http_optimization_gateway import (
    HttpOptimizationGateway,
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


def _request(
    inference_runner: bool | None, fixed_endpoints: bool
) -> OptimizationRequestDto:
    worker_id = ServiceId()
    return OptimizationRequestDto(
        flow_id=FlowId(user_id=UserId()),
        task=ModelTask.CLASSIFICATION,
        stream_info=StreamInfoDto(arrival_rate=0.1),
        qos_requirements=QoSRequirementsDto(
            max_mean_response_time=10,
            min_model_performance=PerformanceMetric(
                PerformanceMetricType.MAP_AT_50, "test", 0.8
            ),
            max_energy_consumption=100,
        ),
        flow_endpoint_placement=FlowEndpointPlacementDto(
            ingress_placement=FixedPlacementDto(worker_id=worker_id)
            if fixed_endpoints
            else AnyPlacementDto(),
            sink_placement=AnyPlacementDto(),
        ),
        local_worker_config=(
            NoLocalWorkerConfigDto()
            if inference_runner is None
            else ValidLocalWorkerConfigDto(
                worker_id=worker_id, is_inference_runner=inference_runner
            )
        ),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("inference_runner", [True, False, None])
@pytest.mark.parametrize("fixed_endpoints", [True, False])
async def test_sends_new_protocol_without_overriding_flow_constraints(
    inference_runner: bool | None, fixed_endpoints: bool
) -> None:
    requests = (_request(inference_runner, fixed_endpoints), _request(None, False))
    sent: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        assert request.method == "POST"
        assert str(request.url) == "http://controller.local/controller/optimization"
        decoded = OptimizeInferenceDistributionRequest.model_validate_json(
            request.content
        )
        assert decoded.requests == requests
        assert request.extensions["timeout"]["read"] == 300
        return httpx.Response(204)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        await HttpOptimizationGateway(
            client, "http://controller.local/"
        ).optimize_inference_distribution(requests)
    assert len(sent) == 1


@pytest.mark.asyncio
async def test_propagates_controller_failure() -> None:
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(503))
    ) as client:
        gateway = HttpOptimizationGateway(client, "http://controller.local")
        with pytest.raises(httpx.HTTPStatusError):
            await gateway.optimize_inference_distribution((_request(False, True),))


@pytest.mark.asyncio
@pytest.mark.parametrize("duplicate", [True, False])
async def test_rejects_duplicate_flows_or_empty_batch_before_sending(
    duplicate: bool,
) -> None:
    sent: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(204)

    dto = _request(False, True)
    requests = (dto, dto) if duplicate else ()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(ValueError):
            await HttpOptimizationGateway(
                client, "http://controller.local"
            ).optimize_inference_distribution(requests)
    assert sent == []
