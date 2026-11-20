import asyncio
import json
from pathlib import Path
from uuid import UUID

import httpx
import numpy as np
import onnx
import pytest
from test.system import test_single_flow_traffic as traffic_test
from test.system.inference_traffic import run_inference_traffic
from test.system.optimization_scenario import OptimizationScenario

from integration.dto.flow_plan.flow_plan_dto import (
    FlowPlanDto,
    InputSpecDto,
    ModelSpecsDto,
)
from integration.dto.tensor.tensor_bundle_dto import TensorBundleDto
from integration.front_end.http.front_end_http_schema import RunInferenceRequest
from shared.flow.flow import FlowId
from shared.model.model import ModelId
from shared.model.model_variant import ModelVariantId
from shared.plan.hop_logical_id import PlannedIngressReplicaId, PlannedSinkReplicaId
from shared.plan.plan_version import PlanVersion
from shared.service.service import ServiceId


@pytest.fixture
def request_body() -> RunInferenceRequest:
    return RunInferenceRequest(
        flow_id=FlowId(flow_id=UUID(int=1)),
        bundle=TensorBundleDto.from_ndarrays(
            {"input": np.array([1.0], dtype="float32")}
        ),
    )


@pytest.mark.asyncio
async def test_arrivals_continue_while_all_previous_responses_are_pending(
    request_body: RunInferenceRequest,
) -> None:
    all_arrived = asyncio.Event()
    requests = []

    async def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 3:
            all_arrived.set()
        await all_arrived.wait()
        return httpx.Response(200, json={"bundle": request_body.bundle.model_dump()})

    async with httpx.AsyncClient(
        base_url="http://front-end", transport=httpx.MockTransport(handle)
    ) as client:
        samples = await asyncio.wait_for(
            run_inference_traffic(
                client, request_body, arrival_rate=100.0, request_count=3
            ),
            timeout=2.0,
        )

    assert len(requests) == len(samples) == 3
    assert [sample.scheduled_at_s for sample in samples] == [0.0, 0.01, 0.02]
    assert [sample.request_number for sample in samples] == [1, 2, 3]
    assert (
        samples[0].response_time_s >= samples[2].started_at_s - samples[0].started_at_s
    )
    assert all(
        sample.response is not None and sample.error is None for sample in samples
    )
    assert all(
        json.loads(request.content) == request_body.model_dump(mode="json")
        for request in requests
    )


@pytest.mark.asyncio
async def test_http_errors_and_timeouts_do_not_stop_later_arrivals(
    request_body: RunInferenceRequest,
) -> None:
    calls = 0

    async def handle(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ReadTimeout("request expired", request=request)
        return httpx.Response(500 if calls == 2 else 200)

    async with httpx.AsyncClient(
        base_url="http://front-end", transport=httpx.MockTransport(handle)
    ) as client:
        samples = await run_inference_traffic(
            client, request_body, arrival_rate=100.0, request_count=3
        )
    assert calls == 3
    assert samples[0].response is None
    assert samples[0].error == "ReadTimeout: request expired"
    assert samples[1].response is not None
    assert samples[1].response.status_code == 500
    assert samples[2].response is not None
    assert samples[2].response.status_code == 200


@pytest.mark.asyncio
@pytest.mark.parametrize("arrival_rate", [0.0, -1.0, float("nan"), float("inf")])
async def test_invalid_rate_is_rejected_before_sending(
    request_body: RunInferenceRequest, arrival_rate: float
) -> None:
    async with httpx.AsyncClient(base_url="http://front-end") as client:
        with pytest.raises(ValueError, match="arrival_rate"):
            await run_inference_traffic(
                client, request_body, arrival_rate=arrival_rate, request_count=2
            )


@pytest.mark.parametrize("failure", [None, "wrong_output", "http_error"])
def test_single_flow_scenario_uses_same_rate_and_validates_every_response(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    failure: str | None,
) -> None:
    model_path = tmp_path / "identity.onnx"
    model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [onnx.helper.make_node("Identity", ["input"], ["output"])],
            "identity",
            [onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1])],
            [onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1])],
        ),
        opset_imports=[onnx.helper.make_opsetid("", 13)],
        ir_version=10,
    )
    onnx.save(model, model_path)
    flow = FlowId(flow_id=UUID(int=1))
    variant = ModelVariantId(ModelId(model_name="identity"), "fp32")
    worker = ServiceId(id="worker")
    plan = FlowPlanDto(
        plan_version=PlanVersion(version_number=1),
        model_specs=ModelSpecsDto(
            model_variant_id=variant,
            inputs=(InputSpecDto(name="input", shape=(1,), dtype="float32"),),
        ),
        ingress_replica_id=PlannedIngressReplicaId(variant, replica_idx=1),
        ingress_worker_id=worker,
        sink_replica_id=PlannedSinkReplicaId(variant, replica_idx=2),
        sink_worker_id=worker,
    )

    def prepare(
        workspace: Path,
        *,
        arrival_rates: tuple[float, ...],
        max_response_times_s: tuple[float, ...],
    ) -> OptimizationScenario:
        assert workspace == tmp_path
        assert arrival_rates == (100.0,)
        assert max_response_times_s == (100.0,)
        return OptimizationScenario(
            "http://front-end", model_path, {flow: plan}, frozenset({variant}), worker
        )

    calls = 0

    async def handle(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        body = RunInferenceRequest.model_validate_json(request.content)
        assert body.flow_id == flow
        if failure == "http_error":
            return httpx.Response(500, text="inference failed")
        output = body.bundle.to_ndarrays()["input"]
        if failure == "wrong_output":
            output = output + 1
        bundle = TensorBundleDto.from_ndarrays({"output": output})
        return httpx.Response(200, json={"bundle": bundle.model_dump(mode="json")})

    async_client = httpx.AsyncClient
    monkeypatch.setenv("TRAFFIC_ARRIVAL_RATE", "100")
    monkeypatch.setenv("TRAFFIC_REQUEST_COUNT", "3")
    monkeypatch.setenv("TRAFFIC_MAX_RESPONSE_TIME_S", "100")
    monkeypatch.setenv("TRAFFIC_MAX_START_LAG_S", "1")
    monkeypatch.setattr(traffic_test, "prepare_optimization_scenario", prepare)
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: async_client(transport=httpx.MockTransport(handle), **kwargs),
    )
    if failure is None:
        traffic_test.test_single_flow_traffic(tmp_path, capsys)
    else:
        with pytest.raises(AssertionError, match="Request 1"):
            traffic_test.test_single_flow_traffic(tmp_path, capsys)
    assert calls == 3
