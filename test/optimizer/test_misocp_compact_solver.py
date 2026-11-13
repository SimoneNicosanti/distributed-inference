from dataclasses import replace
from uuid import UUID

import gurobipy as gp
import pytest
from test.optimizer.test_misocp_compact_formulation_builder import _build, _problem

from optimizer.adapters.outbound.gurobi.entities.gurobi_solver_config import (
    GurobiSolverConfig,
)
from optimizer.adapters.outbound.gurobi.formulation_compiler.formulations.compact.gurobi_misocp_compact_formulation_compiler import (
    GurobiMisocpCompactFormulationModelBuilder,
)
from optimizer.adapters.outbound.gurobi.formulation_compiler.gurobi_formulation_compiler import (
    GurobiModelBuilder,
)
from optimizer.adapters.outbound.gurobi.result_translation.formulations.compact.gurobi_misocp_compact_solved_compiled_formulation_translator import (
    GurobiCompactSolvedFormulationModelTranslator,
)
from optimizer.adapters.outbound.gurobi.solver.gurobi_solver import GurobiSolver
from optimizer.adapters.outbound.gurobi.warm_start.compact_gurobi_warm_start_applier import (
    CompactGurobiWarmStartApplier,
)
from optimizer.application.ports.outbound.solver_result import SolverTerminationKind
from optimizer.domain.problem.other.backend import Backend
from optimizer.domain.problem.other.flow_endpoint_placement import (
    FixedPlacement,
    FlowEndpointPlacement,
)
from optimizer.domain.problem.other.network_connection import NetworkConnection
from shared.service.service import ServiceId


@pytest.fixture(scope="module", autouse=True)
def _license():
    try:
        with gp.Env(params={"OutputFlag": 0}):
            pass
    except gp.GurobiError as error:
        pytest.skip(f"Gurobi environment unavailable: {error}")


@pytest.mark.asyncio
async def test_selects_feasible_shape_and_translates_it_through_native_solver():
    problem = _problem()
    flow = next(iter(problem.flows))
    problem = replace(
        problem, flows={flow: replace(problem.flows[flow], arrival_rate_per_s=10.0)}
    )
    formulation = _build(problem)
    builder = GurobiModelBuilder(
        GurobiMisocpCompactFormulationModelBuilder(), non_convex=0
    )
    async with builder.build_model_from_formulation(formulation) as model:
        result = await GurobiSolver().solve_compiled_formulation(
            model, GurobiSolverConfig(time_limit_s=10, threads=1)
        )
        assert result.termination is SolverTerminationKind.OPTIMAL
        assert result.solved_model is not None
        solution = await GurobiCompactSolvedFormulationModelTranslator().translate_solved_formulation_model(
            result.solved_model
        )
        selected = solution.flow_solutions[0]
        assert selected.input_shapes == {"x": (4, 3)}
        assert selected.model_variant_id == next(iter(problem.variants))
        assert selected.mean_batch_fill_time_s == pytest.approx(0.15)
        assert selected.components[0].tensor_contract.input_tensor_keys == ("x",)
        assert selected.components[0].tensor_contract.output_tensor_keys == ("y",)
        assert selected.components[0].backend is Backend.CPU
        await CompactGurobiWarmStartApplier().apply_warm_start(model, solution)
        assert (
            sum(
                variable.Start
                for variable in model.variables.assignment.configuration_selected.values()
            )
            == 1
        )


@pytest.mark.asyncio
async def test_directed_network_with_fixed_endpoints_and_gpu_uses_normalized_costs():
    problem = _problem()
    flow = next(iter(problem.flows))
    source = next(iter(problem.workers))
    target = ServiceId(UUID(int=4))
    problem = replace(
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
    formulation = _build(problem)
    async with GurobiModelBuilder(
        GurobiMisocpCompactFormulationModelBuilder(), non_convex=0
    ).build_model_from_formulation(formulation) as model:
        result = await GurobiSolver().solve_compiled_formulation(
            model, GurobiSolverConfig(time_limit_s=10, threads=1)
        )
        assert result.termination is SolverTerminationKind.OPTIMAL
        solution = await GurobiCompactSolvedFormulationModelTranslator().translate_solved_formulation_model(
            model
        )
        selected = solution.flow_solutions[0]
        assert selected.ingress_worker_id == source
        assert selected.sink_worker_id == target
        assert selected.components[0].executable_layer_keys == ()
        assert selected.components[1].executable_layer_keys == ("compute",)
        assert selected.components[1].backend is Backend.GPU
        assert selected.cuts[0].tensor_keys == ("x",)
        assert selected.components[0].transmission_latency_s == pytest.approx(0.01)


@pytest.mark.asyncio
@pytest.mark.parametrize("unsupported", ["memory", "execution"])
async def test_missing_execution_and_insufficient_gpu_memory_remain_infeasible(
    unsupported,
):
    problem = _problem()
    worker = next(iter(problem.workers))
    problem = replace(
        problem,
        workers={
            worker: replace(
                problem.workers[worker],
                backend=Backend.GPU,
                available_inference_memory_bytes=1
                if unsupported == "memory"
                else 100_000_000,
            )
        },
    )
    if unsupported == "execution":
        problem = replace(
            problem,
            execution_times_s={
                (configuration, worker): {"input": 0.0, "output": 0.0}
                for configuration in problem.configurations
            },
        )
    async with GurobiModelBuilder(
        GurobiMisocpCompactFormulationModelBuilder(), non_convex=0
    ).build_model_from_formulation(_build(problem)) as model:
        result = await GurobiSolver().solve_compiled_formulation(
            model, GurobiSolverConfig(time_limit_s=10, threads=1)
        )
        assert result.termination is SolverTerminationKind.INFEASIBLE
        assert result.solved_model is None
