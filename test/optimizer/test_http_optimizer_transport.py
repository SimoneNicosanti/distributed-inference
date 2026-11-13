import json
from collections.abc import Callable
from dataclasses import replace
from typing import Any, override

import httpx
import pytest
from fastapi import FastAPI
from pydantic import ValidationError
from test.controller.application.optimization.test_compact_optimization_problem_factory import (
    _context,
    _factory,
    _request,
    _worker,
)

from controller.adapters.outbound.optimizer.http_optimizer_gateway import (
    HttpOptimizerGateway,
)
from controller.adapters.outbound.optimizer.mappers.optimization_problem_mapper import (
    OptimizationProblemMapper as ControllerProblemMapper,
)
from controller.adapters.outbound.optimizer.mappers.optimization_solution_mapper import (
    OptimizationSolutionMapper as ControllerSolutionMapper,
)
from controller.domain.backend import Backend as ControllerBackend
from controller.domain.optimization.problem.compact_optimization_problem import (
    CompactOptimizationProblem as ControllerProblem,
)
from controller.domain.optimization.problem.optimization_problem import (
    OptimizationProblem as ControllerProblemBase,
)
from controller.domain.optimization.solution.optimization_solution import (
    OptimizationSolution as ControllerSolution,
)
from integration.optimizer.http.optimizer_http_paths import OPTIMIZE_PATH
from integration.optimizer.http.optimizer_http_schema import (
    OptimizeRequest,
    OptimizeResponse,
)
from optimizer.adapters.inbound.http_optimizer_controller import build_optimizer_router
from optimizer.adapters.inbound.mappers.optimization_problem_mapper import (
    OptimizationProblemMapper as OptimizerProblemMapper,
)
from optimizer.adapters.inbound.mappers.optimization_solution_mapper import (
    OptimizationSolutionMapper as OptimizerSolutionMapper,
)
from optimizer.application.ports.inbound.optimizer_controller import OptimizerController
from optimizer.domain.problem.other.backend import Backend
from optimizer.domain.problem.compact.compact_optimization_problem import (
    CompactOptimizationProblem,
)
from optimizer.domain.problem.optimization_problem import OptimizationProblem
from optimizer.domain.solution.optimization_solution import (
    ComponentSolution,
    ComponentTensorContract,
    CutSolution,
    FlowOptimizationSolution,
    OptimizationSolution,
)
from shared.model.model import ModelId
from shared.model.model_variant import ModelVariantId


async def _problem() -> ControllerProblem:
    request = _request()
    return await _factory().build_optimization_problem(
        {request.flow_id: request}, _context()
    )


def _solution(problem: CompactOptimizationProblem) -> OptimizationSolution:
    flow_id = next(iter(problem.flows))
    configuration = next(
        key for key, data in problem.configurations.items() if data.batch_size == 4
    )
    components = tuple(
        ComponentSolution(
            component_index=index,
            worker_id=_worker(index),
            backend=Backend.GPU if index == 2 else Backend.CPU,
            layer_keys=layers,
            input_layer_keys=("input",) if index == 1 else (),
            output_layer_keys=("output",) if index == 3 else (),
            tensor_contract=ComponentTensorContract(inputs, outputs),
            priority=2,
            mean_execution_time_s=0.12 if index == 2 else 0.0,
            mean_execution_queue_time_s=0.2 if index == 2 else 0.0,
            mean_transmission_time_s=0.01,
            transmission_latency_s=0.02,
        )
        for index, layers, inputs, outputs in (
            (1, ("input",), ("features", "mask"), ("features", "mask")),
            (2, ("a", "b"), ("features", "mask"), ("prediction",)),
            (3, ("output",), ("prediction",), ("prediction",)),
        )
    )
    return OptimizationSolution(
        flow_solutions=(
            FlowOptimizationSolution(
                flow_id=flow_id,
                ingress_worker_id=_worker(1),
                sink_worker_id=_worker(3),
                model_variant_id=configuration.model_variant_id,
                components=components,
                cuts=(
                    CutSolution(1, 2, ("features", "mask")),
                    CutSolution(2, 3, ("prediction",)),
                ),
                mean_batch_fill_time_s=0.75,
                mean_response_time_s=1.1,
                input_shapes=configuration.input_shapes.input_shapes,
            ),
        ),
        objective_value=1.1,
        is_optimal=False,
    )


class StubOptimizer(OptimizerController):
    def __init__(
        self,
        transform: Callable[[OptimizationSolution], OptimizationSolution] | None = None,
    ) -> None:
        self.problems: list[CompactOptimizationProblem] = []
        self.transform = transform

    @override
    async def optimize(self, problem: OptimizationProblem) -> OptimizationSolution:
        assert isinstance(problem, CompactOptimizationProblem)
        self.problems.append(problem)
        solution = _solution(problem)
        return self.transform(solution) if self.transform is not None else solution


def _app(optimizer: OptimizerController) -> FastAPI:
    app = FastAPI()
    app.include_router(build_optimizer_router(optimizer))
    return app


@pytest.mark.asyncio
async def test_controller_gateway_to_optimizer_router_and_back_preserves_the_contract() -> (
    None
):
    original = await _problem()
    optimizer = StubOptimizer()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=_app(optimizer))
    ) as client:
        solution = await HttpOptimizerGateway(
            client, "http://optimizer.test/"
        ).optimize(original)
    assert isinstance(solution, ControllerSolution)
    assert len(optimizer.problems) == 1
    mapped = optimizer.problems[0]
    assert type(mapped) is CompactOptimizationProblem
    assert set(mapped.variants) == set(original.variants)
    assert len(mapped.configurations) == 2
    assert len(mapped.flows[next(iter(mapped.flows))].configurations) == 2
    assert {key.input_shapes for key in mapped.configurations} != set()
    assert mapped.workers[_worker(2)].available_inference_memory_bytes == 96
    assert mapped.workers[_worker(2)].backend is Backend.GPU
    assert mapped.connections[0].throughput_bytes_per_sec == 1000
    assert mapped.policy.objective.value == original.policy.objective.value
    returned = solution.flow_solutions[0]
    assert returned.flow_id in original.flows
    assert returned.ingress_worker_id == _worker(1)
    assert returned.sink_worker_id == _worker(3)
    assert returned.components[1].backend is ControllerBackend.GPU
    assert returned.components[1].tensor_contract.input_tensor_keys == (
        "features",
        "mask",
    )
    assert returned.cuts[0].tensor_keys == ("features", "mask")
    assert returned.input_shapes == {"features": (4, 3), "mask": (4,)}
    assert solution.objective_value == 1.1
    assert not solution.is_optimal


@pytest.mark.asyncio
async def test_json_round_trip_preserves_every_projected_field() -> None:
    original = await _problem()
    request = OptimizeRequest(problem=ControllerProblemMapper.to_dto(original))
    decoded = OptimizeRequest.model_validate_json(request.model_dump_json())
    mapped = OptimizerProblemMapper.from_dto(decoded.problem)
    assert mapped.variants[next(iter(mapped.variants))].weight_bytes_by_layer == {
        "input": 0,
        "a": 24,
        "b": 0,
        "output": 0,
    }
    for key, costs in original.execution_times_s.items():
        match = next(
            (other, worker)
            for other, worker in mapped.execution_times_s
            if other.model_variant_id == key[0].model_variant_id
            and other.input_shapes.input_shapes == key[0].input_shapes.input_shapes
            and worker == key[1]
        )
        assert mapped.execution_times_s[match] == costs
    source_flow = next(iter(original.flows.values()))
    target_flow = next(iter(mapped.flows.values()))
    assert target_flow.arrival_rate_per_s == source_flow.arrival_rate_per_s
    assert target_flow.max_response_time_s == source_flow.max_response_time_s
    assert target_flow.performance_requirement == source_flow.performance_requirement
    assert target_flow.max_energy_consumption == source_flow.max_energy_consumption
    assert target_flow.endpoint_placement.ingress_placement.worker_id == _worker(1)
    assert target_flow.local_worker_config.worker_id == _worker(1)
    assert not target_flow.local_worker_config.is_inference_runner
    result = _solution(mapped)
    envelope = OptimizeResponse(solution=OptimizerSolutionMapper.to_dto(result))
    parsed = OptimizeResponse.model_validate_json(envelope.model_dump_json())
    restored = ControllerSolutionMapper.from_dto(parsed.solution)
    assert (
        restored.flow_solutions[0].input_shapes == result.flow_solutions[0].input_shapes
    )


def _invalid_payload(payload: dict[str, Any], failure: str) -> None:
    problem = payload["problem"]
    if failure == "duplicate_variant":
        problem["variants"].append(problem["variants"][0])
    elif failure == "duplicate_configuration":
        problem["configurations"].append(problem["configurations"][0])
    elif failure == "duplicate_worker":
        problem["workers"].append(problem["workers"][0])
    elif failure == "duplicate_flow":
        problem["flows"].append(problem["flows"][0])
    elif failure == "duplicate_execution":
        problem["execution_times_s"].append(problem["execution_times_s"][0])
    elif failure == "duplicate_connection":
        problem["connections"].append(problem["connections"][0])
    elif failure == "unknown_configuration":
        problem["flows"][0]["configurations"][0]["input_shapes"]["input_shapes"][
            "features"
        ][0] = 7
    elif failure == "unknown_cost_layer":
        problem["execution_times_s"][0]["layer_times_s"]["unknown"] = 1.0
    elif failure == "missing_tensor":
        del problem["configurations"][0]["tensor_bytes"]["hidden"]
    elif failure == "wrong_batch":
        problem["configurations"][0]["batch_size"] = 9
    elif failure == "negative_time":
        problem["execution_times_s"][0]["layer_times_s"]["a"] = -1.0
    elif failure == "boolean_memory":
        problem["workers"][0]["available_inference_memory_bytes"] = True
    elif failure == "missing_worker_id":
        problem["workers"][0]["worker_id"] = {}
    elif failure == "missing_flow_id":
        problem["flows"][0]["flow_id"] = {
            "user_id": {"id": "00000000-0000-0000-0000-000000000001"}
        }
    elif failure == "missing_user_id":
        problem["flows"][0]["flow_id"]["user_id"] = {}
    elif failure == "unsupported_kind":
        problem["kind"] = "unknown"
    elif failure == "raw_context":
        problem["context"] = {}
    elif failure == "cyclic_graph":
        problem["variants"][0]["edges"].append(
            {"source": "b", "target": "a", "tensors": []}
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [
        "duplicate_variant",
        "duplicate_configuration",
        "duplicate_worker",
        "duplicate_flow",
        "duplicate_execution",
        "duplicate_connection",
        "unknown_configuration",
        "unknown_cost_layer",
        "missing_tensor",
        "wrong_batch",
        "negative_time",
        "boolean_memory",
        "missing_worker_id",
        "missing_flow_id",
        "missing_user_id",
        "unsupported_kind",
        "raw_context",
        "cyclic_graph",
    ],
)
async def test_rejects_invalid_payload_before_calling_optimizer(failure: str) -> None:
    optimizer = StubOptimizer()
    request = OptimizeRequest(problem=ControllerProblemMapper.to_dto(await _problem()))
    payload = json.loads(request.model_dump_json())
    _invalid_payload(payload, failure)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=_app(optimizer)),
        base_url="http://optimizer.test",
    ) as client:
        response = await client.post(OPTIMIZE_PATH, json=payload)
    assert response.status_code == 422
    assert optimizer.problems == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [
        "missing_shape",
        "wrong_variant",
        "wrong_flows",
        "wrong_backend",
        "fixed_endpoint",
    ],
)
async def test_does_not_publish_a_solution_for_a_different_problem(
    failure: str,
) -> None:
    def transform(solution: OptimizationSolution) -> OptimizationSolution:
        flow = solution.flow_solutions[0]
        if failure == "missing_shape":
            flow = replace(flow, input_shapes=None)
        elif failure == "wrong_variant":
            flow = replace(
                flow, model_variant_id=ModelVariantId(ModelId("wrong"), "variant")
            )
        elif failure == "wrong_flows":
            return replace(solution, flow_solutions=())
        elif failure == "wrong_backend":
            components = list(flow.components)
            components[1] = replace(components[1], backend=Backend.CPU)
            flow = replace(flow, components=tuple(components))
        else:
            components = list(flow.components)
            components[0] = replace(components[0], worker_id=_worker(3))
            flow = replace(
                flow, ingress_worker_id=_worker(3), components=tuple(components)
            )
        return replace(solution, flow_solutions=(flow,))

    optimizer = StubOptimizer(transform)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=_app(optimizer), raise_app_exceptions=False)
    ) as client:
        with pytest.raises(httpx.HTTPStatusError) as caught:
            await HttpOptimizerGateway(client, "http://optimizer.test").optimize(
                await _problem()
            )
    assert caught.value.response.status_code == 500
    assert len(optimizer.problems) == 1


@pytest.mark.asyncio
async def test_gateway_rejects_unrelated_response_from_an_external_server() -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == OPTIMIZE_PATH
        assert request.extensions["timeout"]["read"] == 45.0
        incoming = OptimizeRequest.model_validate_json(request.content)
        local_problem = OptimizerProblemMapper.from_dto(incoming.problem)
        dto = OptimizerSolutionMapper.to_dto(
            replace(_solution(local_problem), flow_solutions=())
        )
        return httpx.Response(
            200, json=OptimizeResponse(solution=dto).model_dump(mode="json")
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        with pytest.raises(ValueError, match="different set of flows"):
            await HttpOptimizerGateway(
                client, "http://optimizer.test/", timeout_s=45.0
            ).optimize(await _problem())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure", ["http", "timeout", "malformed_response", "unsupported_problem"]
)
async def test_gateway_propagates_transport_and_contract_errors(failure: str) -> None:
    calls = []

    def respond(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if failure == "timeout":
            raise httpx.ReadTimeout("optimizer timed out", request=request)
        if failure == "http":
            return httpx.Response(503)
        return httpx.Response(200, json={"solution": {}})

    expected = {
        "http": httpx.HTTPStatusError,
        "timeout": httpx.ReadTimeout,
        "malformed_response": ValidationError,
        "unsupported_problem": TypeError,
    }[failure]
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        gateway = HttpOptimizerGateway(client, "http://optimizer.test")
        problem = (
            ControllerProblemBase()
            if failure == "unsupported_problem"
            else await _problem()
        )
        with pytest.raises(expected):
            await gateway.optimize(problem)
    assert len(calls) == (0 if failure == "unsupported_problem" else 1)
