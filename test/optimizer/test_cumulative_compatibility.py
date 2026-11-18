"""Physical costs, expanded MIP starts and solver observation compatibility."""

from dataclasses import replace
from unittest.mock import Mock

import httpx
import pytest
from fastapi import FastAPI
from gurobipy import GRB
from test.optimizer.test_cumulative_formulation import Kind, dag_problem, formulation
from test.optimizer.test_full_optimizer_runtime import assert_solution, compiler
from test.optimizer.test_single_configuration_contract import controller_problem

from controller.adapters.outbound.optimizer.http_optimizer_gateway import (
    HttpOptimizerGateway,
)
from optimizer.adapters.inbound.http_optimizer_controller import build_optimizer_router
from optimizer.adapters.outbound.gurobi.entities.gurobi_solver_config import (
    GurobiSolverConfig,
)
from optimizer.adapters.outbound.gurobi.result_translation.formulations.compact.gurobi_misocp_compact_solved_compiled_formulation_translator import (
    GurobiMisocpCompactSolvedCompiledFormulationTranslator,
)
from optimizer.adapters.outbound.gurobi.solver.gurobi_solver import GurobiSolver
from optimizer.adapters.outbound.gurobi.trackers.gurobi_progress_tracker import (
    GurobiProgressTracker,
)
from optimizer.adapters.outbound.gurobi.warm_start.compact_gurobi_warm_start_applier import (
    CompactGurobiWarmStartApplier,
)
from optimizer.application.formulation.compact.misocp.misocp_compact_formulation_builder import (
    MisocpCompactFormulationBuilder,
)
from optimizer.application.optimization_controller import DefaultOptimizerController
from optimizer.application.ports.outbound.solver_result import SolverTerminationKind
from optimizer.application.strategies.full_solver_strategy import FullSolverStrategy
from optimizer.domain.formulation.compact.misocp.misocp_compact_formulation_config import (
    MisocpCompactFormulationConfig,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("batch", [1, 4])
async def test_fixed_partition_has_identical_physical_quantities(batch):
    problem = dag_problem()
    selected = next(
        key for key, data in problem.configurations.items() if data.batch_size == batch
    )
    problem = replace(
        problem,
        variants={
            selected.model_variant_id: problem.variants[selected.model_variant_id]
        },
        configurations={selected: problem.configurations[selected]},
        execution_times_s={
            key: times
            for key, times in problem.execution_times_s.items()
            if key[0] == selected
        },
        flows={
            flow: replace(request, configurations=(selected,), max_response_time_s=100)
            for flow, request in problem.flows.items()
        },
    )
    positions = {
        "input": 1,
        "a": 1,
        "b": 1,
        "c": 2,
        "e": 2,
        "join": 3,
        "output": 3,
        "output2": 3,
    }
    observed = []
    for kind in Kind:
        data = formulation(problem, kind)
        flow = data.flows[0]
        source, target = data.problem.flows[flow].workers
        async with compiler().compile_formulation(data) as compiled:
            for (
                _,
                _,
                layer,
                component,
            ), term in compiled.variables.assignment.layer_component.items():
                compiled.native_model.addConstr(
                    term == int(positions[layer] == component)
                )
            for (
                _,
                component,
                worker,
            ), term in compiled.variables.assignment.component_worker.items():
                compiled.native_model.addConstr(
                    term == int(worker == (target if component == 2 else source))
                )
            output = await GurobiSolver().solve_compiled_formulation(
                compiled, GurobiSolverConfig(time_limit_s=5, threads=1, mip_gap=0)
            )
            assert output.termination is SolverTerminationKind.OPTIMAL
            solution = await GurobiMisocpCompactSolvedCompiledFormulationTranslator().translate_solved_formulation_model(
                compiled
            )
            assert_solution(problem, solution)
            result = solution.flow_solutions[0]
            execution = tuple(
                component.mean_execution_time_s for component in result.components
            )
            assert execution == pytest.approx(
                (0.004 * batch, 0.002 * batch, 0.002 * batch)
            )
            cut_sizes = tuple(
                sum(
                    problem.configurations[selected].tensor_bytes[tensor]
                    for tensor in cut.tensor_keys
                )
                for cut in result.cuts
            )
            assert cut_sizes == (96 * batch, 96 * batch)
            observed.append((execution, cut_sizes))
    assert observed[0] == observed[1]


@pytest.mark.asyncio
async def test_small_component_seed_can_initialize_larger_cumulative_model():
    problem = dag_problem(variants=1, workers=2, components=1)
    async with compiler().compile_formulation(
        formulation(problem, Kind.ASSIGNMENT)
    ) as compiled:
        output = await GurobiSolver().solve_compiled_formulation(
            compiled, GurobiSolverConfig(time_limit_s=5, threads=1, mip_gap=0)
        )
        assert output.termination is SolverTerminationKind.OPTIMAL
        seed = await GurobiMisocpCompactSolvedCompiledFormulationTranslator().translate_solved_formulation_model(
            compiled
        )
    problem = replace(problem, policy=replace(problem.policy, max_components=3))
    async with compiler().compile_formulation(
        formulation(problem, Kind.CUMULATIVE)
    ) as compiled:
        await CompactGurobiWarmStartApplier().apply_warm_start(compiled, seed)
        assert all(
            var.Start == 1
            for var in compiled.variables.assignment.layer_decision_variables()
        )
        output = await GurobiSolver().solve_compiled_formulation(
            compiled, GurobiSolverConfig(time_limit_s=5, threads=1, mip_gap=0)
        )
        assert output.termination is SolverTerminationKind.OPTIMAL
        result = await GurobiMisocpCompactSolvedCompiledFormulationTranslator().translate_solved_formulation_model(
            compiled
        )
        assert_solution(problem, result)


def test_progress_observation_sampling_and_zero_objective():
    tracker = GurobiProgressTracker(minimize=True)
    model = Mock()
    model.cbGet.side_effect = [2, 3]
    tracker(model, GRB.Callback.PRESOLVE)
    assert tracker.presolve_columns_removed == 2 and tracker.presolve_rows_removed == 3
    model.cbGet.side_effect = [0.1, 1.0, 0.0]
    tracker(model, GRB.Callback.MIP)
    model.cbGet.side_effect = [0.9]
    tracker(model, GRB.Callback.MIP)
    model.cbGet.side_effect = [1.0, 2.0, 5.0]
    tracker(model, GRB.Callback.MIP)
    assert [point.objective_bound for point in tracker.bound_history] == [1.0, 2.0]
    model.cbGet.side_effect = [0.0, 1.5]
    tracker(model, GRB.Callback.MIPSOL)
    assert tracker.time_to_zero_objective_s == 1.5


@pytest.mark.asyncio
async def test_http_contract_is_unchanged_with_cumulative_optimizer():
    problem = await controller_problem()
    problem = replace(
        problem, policy=replace(problem.policy, max_components=2, priority_count=1)
    )
    strategy = FullSolverStrategy(
        formulation_builder=MisocpCompactFormulationBuilder(),
        formulation_config=MisocpCompactFormulationConfig(0.5, Kind.CUMULATIVE),
        formulation_compiler=compiler(),
        solver_config=GurobiSolverConfig(time_limit_s=5, threads=1, mip_gap=0),
        solver=GurobiSolver(),
        solved_model_translator=GurobiMisocpCompactSolvedCompiledFormulationTranslator(),
    )
    app = FastAPI()
    app.include_router(build_optimizer_router(DefaultOptimizerController(strategy)))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app)) as client:
        solution = await HttpOptimizerGateway(
            client, "http://optimizer.test/"
        ).optimize(problem)
    assert solution.is_optimal
    assert solution.flow_solutions[0].input_shapes == {"features": (1, 3), "mask": (1,)}
