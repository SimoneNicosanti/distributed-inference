"""Dynamic checks for the current full pipeline; legacy tests stay untouched."""

import asyncio
from contextlib import asynccontextmanager, suppress
from dataclasses import replace
from threading import Event
from uuid import UUID

import gurobipy as gp
import httpx
import pytest
from fastapi import FastAPI
from test.optimizer.test_misocp_compact_formulation_builder import _problem

from optimizer.adapters.inbound.http_optimizer_controller import build_optimizer_router
from optimizer.adapters.outbound.gurobi.entities.gurobi_solver_config import (
    GurobiSolverConfig,
)
from optimizer.adapters.outbound.gurobi.formulation_compiler.formulations.compact.compact_variables import (
    CompactVariables,
)
from optimizer.adapters.outbound.gurobi.formulation_compiler.formulations.compact.gurobi_misocp_compact_formulation_compiler import (
    GurobiMisocpCompactFormulationModelBuilder,
)
from optimizer.adapters.outbound.gurobi.formulation_compiler.gurobi_formulation_compiler import (
    GurobiFormulationCompiler,
)
from optimizer.adapters.outbound.gurobi.result_translation.formulations.compact.gurobi_misocp_compact_solved_compiled_formulation_translator import (
    GurobiMisocpCompactSolvedCompiledFormulationTranslator,
)
from optimizer.adapters.outbound.gurobi.solver.gurobi_solver import GurobiSolver
from optimizer.application.abc.solver_strategy import SolveError
from optimizer.application.formulation.compact.misocp.misocp_compact_formulation_builder import (
    MisocpCompactFormulationBuilder,
)
from optimizer.application.optimization_controller import DefaultOptimizerController
from optimizer.application.ports.outbound.solver_result import SolverTerminationKind
from optimizer.application.strategies.full_solver_strategy import FullSolverStrategy
from optimizer.domain.formulation.compact.misocp.misocp_compact_formulation import (
    MisocpCompactFormulation,
)
from optimizer.domain.formulation.compact.misocp.misocp_compact_formulation_config import (
    MisocpCompactFormulationConfig,
)
from optimizer.domain.problem.compact.compact_optimization_problem import ObjectiveKind
from optimizer.domain.problem.other.backend import Backend
from optimizer.domain.problem.other.edge_description import EdgeDescription
from optimizer.domain.problem.other.flow_endpoint_placement import (
    FixedPlacement,
    FlowEndpointPlacement,
)
from optimizer.domain.problem.other.network_connection import NetworkConnection
from shared.service.service import ServiceId


class RecordingSolver(GurobiSolver[MisocpCompactFormulation, CompactVariables]):
    output = None

    async def solve_compiled_formulation(self, compiled_formulation, solver_config):
        self.output = await super().solve_compiled_formulation(
            compiled_formulation, solver_config
        )
        return self.output


def compiler():
    return GurobiFormulationCompiler(
        GurobiMisocpCompactFormulationModelBuilder(), non_convex=0
    )


def pipeline(*, model_compiler=None, solver=None, translator=None):
    return FullSolverStrategy(
        formulation_builder=MisocpCompactFormulationBuilder(),
        formulation_config=MisocpCompactFormulationConfig(0.5),
        formulation_compiler=compiler() if model_compiler is None else model_compiler,
        solver_config=GurobiSolverConfig(time_limit_s=10, threads=1, seed=0, mip_gap=0),
        solver=GurobiSolver[MisocpCompactFormulation, CompactVariables]()
        if solver is None
        else solver,
        solved_model_translator=GurobiMisocpCompactSolvedCompiledFormulationTranslator()
        if translator is None
        else translator,
    )


def assert_solution(problem, solution):
    assert solution.is_optimal
    assert {flow.flow_id for flow in solution.flow_solutions} == set(problem.flows)
    for flow in solution.flow_solutions:
        selected = [
            key
            for key in problem.flows[flow.flow_id].configurations
            if key.model_variant_id == flow.model_variant_id
            and key.input_shapes.input_shapes == flow.input_shapes
        ]
        assert len(selected) == 1
        variant = problem.variants[flow.model_variant_id]
        layers = [
            layer for component in flow.components for layer in component.layer_keys
        ]
        assert len(layers) == len(set(layers)) == len(variant.layers)
        assert set(layers) == set(variant.layers)
        assert len(flow.cuts) == len(flow.components) - 1
        for component in flow.components:
            assert component.worker_id in problem.flows[flow.flow_id].workers
            assert component.backend is problem.workers[component.worker_id].backend
            assert 1 <= component.priority <= problem.policy.priority_count
            times = problem.execution_times_s.get(
                (selected[0], component.worker_id), {}
            )
            assert all(layer in times for layer in component.executable_layer_keys)
            expected_execution = sum(times[layer] for layer in component.layer_keys)
            assert component.mean_execution_time_s == pytest.approx(
                expected_execution, abs=1e-5
            )
        total = flow.mean_batch_fill_time_s + sum(
            component.mean_execution_time_s
            + component.mean_execution_queue_time_s
            + component.mean_transmission_time_s
            + component.transmission_latency_s
            for component in flow.components
        )
        assert flow.mean_response_time_s == pytest.approx(total, abs=1e-5)


@pytest.mark.asyncio
@pytest.mark.parametrize("objective", list(ObjectiveKind))
async def test_all_objectives_through_full_pipeline(objective):
    problem = _problem()
    problem = replace(problem, policy=replace(problem.policy, objective=objective))
    result = await DefaultOptimizerController(pipeline()).optimize(problem)
    if objective is ObjectiveKind.ARRIVAL_WEIGHTED_ACCURACY:
        assert result.objective_value == pytest.approx(0.9)
    elif objective in {
        ObjectiveKind.CONSTANT,
        ObjectiveKind.DEADLINE_VIOLATIONS,
        ObjectiveKind.ARRIVAL_WEIGHTED_NORMALIZED_DELAY,
    }:
        assert result.objective_value == pytest.approx(0)
    assert_solution(problem, result)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "arrival,batch,response", [(1.0, 1, 0.1055555555556), (10.0, 4, 0.45)]
)
async def test_shape_selection_and_response_match_analytical_single_worker(
    arrival, batch, response
):
    problem = _problem()
    flow = next(iter(problem.flows))
    problem = replace(
        problem, flows={flow: replace(problem.flows[flow], arrival_rate_per_s=arrival)}
    )
    result = await pipeline().solve(problem)
    assert_solution(problem, result)
    assert result.flow_solutions[0].input_shapes == {"x": (batch, 3)}
    assert result.objective_value == pytest.approx(response, abs=1e-5)


def network_problem():
    problem = _problem()
    flow = next(iter(problem.flows))
    source = next(iter(problem.workers))
    target = ServiceId(UUID(int=4))
    return replace(
        problem,
        workers={
            source: problem.workers[source],
            target: replace(problem.workers[source], backend=Backend.GPU),
        },
        flows={
            flow: replace(
                problem.flows[flow],
                workers=(source, target),
                endpoint_placement=FlowEndpointPlacement(
                    FixedPlacement(source), FixedPlacement(target)
                ),
            )
        },
        policy=replace(problem.policy, max_components=2),
        connections=(NetworkConnection(source, target, 1_000_000, 0.02),),
        execution_times_s={
            **{
                (configuration, source): {"input": 0.0, "output": 0.0}
                for configuration in problem.configurations
            },
            **{
                (configuration, target): times
                for (configuration, _), times in problem.execution_times_s.items()
            },
        },
    )


@pytest.mark.asyncio
async def test_directed_network_fixed_endpoints_and_gpu_profile_costs():
    problem = network_problem()
    result = await pipeline().solve(problem)
    assert_solution(problem, result)
    flow = result.flow_solutions[0]
    assert len(flow.components) == 2
    assert flow.components[0].executable_layer_keys == ()
    assert flow.components[1].executable_layer_keys == ("compute",)
    assert flow.components[1].backend is Backend.GPU
    assert flow.cuts[0].tensor_keys == ("x",)
    assert flow.components[0].transmission_latency_s == pytest.approx(0.01)


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["memory", "execution", "missing_link"])
async def test_infeasible_cases_reach_native_solver_and_return_no_incumbent(reason):
    problem = network_problem()
    worker = tuple(problem.workers)[1]
    if reason == "memory":
        problem = replace(
            problem,
            workers={
                **problem.workers,
                worker: replace(
                    problem.workers[worker], available_inference_memory_bytes=1
                ),
            },
        )
    elif reason == "execution":
        problem = replace(
            problem,
            execution_times_s={
                pair: {"input": 0.0, "output": 0.0}
                for pair in problem.execution_times_s
            },
        )
    else:
        problem = replace(problem, connections=())
    solver = RecordingSolver()
    with pytest.raises(SolveError):
        await pipeline(solver=solver).solve(problem)
    assert solver.output.termination is SolverTerminationKind.INFEASIBLE
    assert solver.output.solved_model is None


@pytest.mark.asyncio
async def test_two_flows_use_distinct_priority_classes():
    problem = _problem()
    first = next(iter(problem.flows))
    second = replace(first, flow_id=UUID(int=5))
    worker = next(iter(problem.workers))
    candidate = next(iter(problem.configurations))
    data = replace(
        problem.flows[first], arrival_rate_per_s=0.5, configurations=(candidate,)
    )
    problem = replace(
        problem,
        flows={first: data, second: data},
        policy=replace(problem.policy, priority_count=2),
    )

    class PriorityCompiler:
        @asynccontextmanager
        async def compile_formulation(self, formulation):
            async with compiler().compile_formulation(formulation) as model:
                for flow, priority in ((first, 1), (second, 2)):
                    variable = model.variables.compute_queue.component_priority[
                        flow, 1, worker, priority
                    ]
                    variable.LB = variable.UB = 1
                yield model

    result = await pipeline(model_compiler=PriorityCompiler()).solve(problem)
    assert_solution(problem, result)
    flows = {flow.flow_id: flow for flow in result.flow_solutions}
    assert flows[first].components[0].priority == 1
    assert flows[second].components[0].priority == 2
    assert (
        flows[second].components[0].mean_execution_queue_time_s
        >= flows[first].components[0].mean_execution_queue_time_s
    )


@pytest.mark.asyncio
async def test_incomparable_fanout_crosses_once_and_preserves_tensor_contracts():
    problem = network_problem()
    variant = next(iter(problem.variants))
    source, target = problem.workers
    graph = replace(
        problem.variants[variant],
        layers=("input", "a", "b", "output"),
        edges=(
            EdgeDescription("input", "a", frozenset({"x"})),
            EdgeDescription("input", "b", frozenset({"x"})),
            EdgeDescription("a", "output", frozenset({"ya"})),
            EdgeDescription("b", "output", frozenset({"yb"})),
        ),
        weight_bytes_by_layer={"input": 0, "a": 5, "b": 5, "output": 0},
    )
    problem = replace(
        problem,
        variants={variant: graph},
        configurations={
            key: replace(
                data,
                tensor_bytes={
                    "x": 12 * data.batch_size,
                    "ya": 4 * data.batch_size,
                    "yb": 4 * data.batch_size,
                },
                max_cut_size_bytes=20 * data.batch_size,
            )
            for key, data in problem.configurations.items()
        },
        execution_times_s={
            **{
                (key, source): {"input": 0.0, "a": 0.1 * data.batch_size, "output": 0.0}
                for key, data in problem.configurations.items()
            },
            **{
                (key, target): {"input": 0.0, "b": 0.1 * data.batch_size, "output": 0.0}
                for key, data in problem.configurations.items()
            },
        },
    )
    result = await pipeline().solve(problem)
    assert_solution(problem, result)
    flow = result.flow_solutions[0]
    assert flow.cuts[0].tensor_keys == ("x", "ya")
    assert flow.components[0].executable_layer_keys == ("a",)
    assert flow.components[1].executable_layer_keys == ("b",)
    assert flow.components[1].tensor_contract.input_tensor_keys == ("x", "ya")


@pytest.mark.asyncio
async def test_controller_http_gateway_to_real_full_optimizer_and_back():
    from test.optimizer.test_http_optimizer_transport import (
        _problem as controller_problem,
    )

    from controller.adapters.outbound.optimizer.http_optimizer_gateway import (
        HttpOptimizerGateway,
    )

    problem = await controller_problem()
    problem = replace(
        problem, policy=replace(problem.policy, max_components=2, priority_count=1)
    )
    app = FastAPI()
    app.include_router(build_optimizer_router(DefaultOptimizerController(pipeline())))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app)) as client:
        result = await HttpOptimizerGateway(client, "http://optimizer.test/").optimize(
            problem
        )
    assert result.is_optimal
    flow = result.flow_solutions[0]
    assert flow.flow_id in problem.flows
    assert flow.input_shapes == {"features": (1, 3), "mask": (1,)}
    assert len(flow.components) == 2
    assert flow.components[1].executable_layer_keys == ("a", "b")


@pytest.mark.asyncio
@pytest.mark.parametrize("translation_error", [False, True])
async def test_native_model_is_alive_for_translation_and_disposed_afterwards(
    translation_error,
):
    captured = []

    class Translator(GurobiMisocpCompactSolvedCompiledFormulationTranslator):
        async def translate_solved_formulation_model(self, solved_compiled_formulation):
            captured.append(solved_compiled_formulation.native_model)
            assert captured[0].NumVars > 0
            if translation_error:
                raise RuntimeError("translation failed")
            return await super().translate_solved_formulation_model(
                solved_compiled_formulation
            )

    if translation_error:
        with pytest.raises(RuntimeError, match="translation failed"):
            await pipeline(translator=Translator()).solve(_problem())
    else:
        await pipeline(translator=Translator()).solve(_problem())
    assert len(captured) == 1
    with pytest.raises((gp.GurobiError, AttributeError)):
        _ = captured[0].NumVars


@pytest.mark.asyncio
async def test_cancellation_waits_for_solver_thread_before_disposing_native_model():
    started, release = Event(), Event()
    captured = []

    class BlockingSolver(GurobiSolver[MisocpCompactFormulation, CompactVariables]):
        def _solve_model(self, model, config, cancel_requested):
            captured.append(model.native_model)
            started.set()
            if not release.wait(10):
                raise TimeoutError("test worker was not released")
            return super()._solve_model(model, config, cancel_requested)

    task = asyncio.create_task(pipeline(solver=BlockingSolver()).solve(_problem()))
    try:
        assert await asyncio.to_thread(started.wait, 10)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        assert captured[0].NumVars > 0
    finally:
        release.set()
        with suppress(asyncio.CancelledError):
            await asyncio.wait_for(task, 10)
    with pytest.raises((gp.GurobiError, AttributeError)):
        _ = captured[0].NumVars
