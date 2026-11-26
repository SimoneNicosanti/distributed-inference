"""Native MISOCP checks for partial starts, unrestricted prefixes and shared queues."""

from dataclasses import replace

import pytest
from gurobipy import GRB
from test.optimizer.test_cumulative_formulation import dag_problem, formulation
from test.optimizer.test_full_optimizer_runtime import assert_solution, compiler

from optimizer.adapters.outbound.gurobi.entities.gurobi_solver_config import (
    GurobiSolverConfig,
)
from optimizer.adapters.outbound.gurobi.entities.gurobi_strategy_config import (
    GurobiStrategyConfig,
)
from optimizer.adapters.outbound.gurobi.result_translation.formulations.compact.gurobi_misocp_compact_solved_compiled_formulation_translator import (
    GurobiMisocpCompactSolvedCompiledFormulationTranslator,
)
from optimizer.adapters.outbound.gurobi.solver.gurobi_solver import GurobiSolver
from optimizer.adapters.outbound.gurobi.strategy_factory import (
    build_compact_gurobi_solver_strategy,
)
from optimizer.adapters.outbound.gurobi.warm_start.compact_gurobi_incremental_model_adapter import (
    CompactGurobiIncrementalModelAdapter,
)
from optimizer.adapters.outbound.gurobi.warm_start.compact_gurobi_warm_start_applier import (
    CompactGurobiWarmStartApplier,
)
from optimizer.adapters.outbound.gurobi.warm_start.compact_solution_decisions import (
    compact_solution_decisions,
)
from optimizer.application.ports.outbound.solver_result import (
    SolverTerminationKind as Termination,
)
from optimizer.application.strategies.full_solver_strategy import FullSolverStrategy
from optimizer.application.strategies.incremental.compact_problem_prefix import (
    compact_problem_prefix,
)
from optimizer.application.strategies.incremental.incremental_flow_solver_strategy import (
    IncrementalFlowSolverStrategy,
)
from optimizer.application.strategies.solver_strategy_kind import SolverStrategyKind
from optimizer.domain.formulation.compact.misocp.compact_formulation_kind import (
    CompactFormulationKind as Kind,
)
from optimizer.domain.formulation.compact.misocp.misocp_compact_formulation_config import (
    MisocpCompactFormulationConfig,
)
from optimizer.domain.problem.compact.compact_optimization_problem import ObjectiveKind
from optimizer.domain.problem.other.flow_endpoint_placement import (
    AnyPlacement,
    FlowEndpointPlacement,
)

SOLVER_CONFIG = GurobiSolverConfig(
    time_limit_s=5, threads=1, mip_gap=0, seed=0, record_incumbent_history=True
)


async def solve_native(model):
    output = await GurobiSolver().solve_compiled_formulation(model, SOLVER_CONFIG)
    assert output.solved_model is not None, output.termination
    assert CompactGurobiIncrementalModelAdapter().is_valid_solution(model)
    return await GurobiMisocpCompactSolvedCompiledFormulationTranslator().translate_solved_formulation_model(
        model
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", list(Kind))
async def test_partial_start_preserves_all_bounds_and_only_sets_prior_decisions(kind):
    problem = dag_problem(variants=2, flows=2, workers=1, components=2)
    prefix = compact_problem_prefix(problem, 1)
    async with compiler().compile_formulation(formulation(prefix, kind)) as model:
        seed = await solve_native(model)
    async with compiler().compile_formulation(formulation(problem, kind)) as model:
        starts = CompactGurobiWarmStartApplier()
        with pytest.raises(ValueError, match="exactly"):
            await starts.apply_warm_start(model, seed)
        assert all(v.Start == GRB.UNDEFINED for v in model.native_model.getVars())
        await starts.apply_partial_warm_start(model, seed)
        decisions = compact_solution_decisions(model, seed, allow_partial=True)
        decision_indices = {v.index for v, _ in decisions}
        assert decisions and all(v.Start == value for v, value in decisions)
        assert all(
            v.Start == GRB.UNDEFINED
            for v in model.native_model.getVars()
            if v.index not in decision_indices
        )
        original_bounds = {v.index: (v.LB, v.UB) for v in model.native_model.getVars()}
        await CompactGurobiIncrementalModelAdapter().initialize_model(model, seed)
        assert all(
            original_bounds[v.index] == (v.LB, v.UB)
            for v in model.native_model.getVars()
        )
        solution = await solve_native(model)
        assert_solution(problem, solution)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", list(Kind))
async def test_malformed_partial_start_is_atomic_and_reapplication_clears_stale_values(
    kind,
):
    problem = dag_problem(variants=1, flows=2)
    async with compiler().compile_formulation(formulation(problem, kind)) as model:
        full = await solve_native(model)
        starts = CompactGurobiWarmStartApplier()
        await starts.apply_warm_start(model, full)
        before = [v.Start for v in model.native_model.getVars()]
        bad_component = replace(full.flow_solutions[-1].components[0], priority=999)
        bad_flow = replace(
            full.flow_solutions[-1],
            components=(bad_component, *full.flow_solutions[-1].components[1:]),
        )
        bad = replace(full, flow_solutions=(full.flow_solutions[0], bad_flow))
        with pytest.raises(ValueError, match="priority"):
            await starts.apply_partial_warm_start(model, bad)
        assert [v.Start for v in model.native_model.getVars()] == before
        partial = replace(full, flow_solutions=(full.flow_solutions[-1],))
        await starts.apply_partial_warm_start(model, partial)
        decisions = compact_solution_decisions(model, partial, allow_partial=True)
        indices = {v.index for v, _ in decisions}
        assert all(v.Start == value for v, value in decisions)
        assert all(
            v.Start == GRB.UNDEFINED
            for v in model.native_model.getVars()
            if v.index not in indices
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", list(Kind))
async def test_insertion_recomputes_queueing_latency_of_previous_flow(kind):
    problem = dag_problem(variants=1, workers=1, components=1, flows=2, priorities=2)
    async with compiler().compile_formulation(
        formulation(compact_problem_prefix(problem, 1), kind)
    ) as model:
        seed = await solve_native(model)
    async with compiler().compile_formulation(formulation(problem, kind)) as model:
        await CompactGurobiIncrementalModelAdapter().initialize_model(model, seed)
        solution = await solve_native(model)
        previous = next(
            f
            for f in solution.flow_solutions
            if f.flow_id == seed.flow_solutions[0].flow_id
        )
        assert (
            previous.components[0].mean_execution_queue_time_s
            > seed.flow_solutions[0].components[0].mean_execution_queue_time_s
        )
        assert (
            previous.mean_response_time_s > seed.flow_solutions[0].mean_response_time_s
        )
        assert_solution(problem, solution)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", list(Kind))
async def test_unrestricted_prefix_moves_previous_flow_without_fixing_or_repair(
    kind,
):
    base = dag_problem(variants=1, workers=2, components=1, flows=2)
    (first, request), (second, _) = base.flows.items()
    slow, fast = request.workers
    problem = replace(
        base,
        flows={
            first: replace(
                request,
                arrival_rate_per_s=40,
                max_response_time_s=100,
                endpoint_placement=FlowEndpointPlacement(
                    AnyPlacement(), AnyPlacement()
                ),
            ),
            second: replace(
                request,
                arrival_rate_per_s=140,
                max_response_time_s=100,
                workers=(fast,),
                endpoint_placement=FlowEndpointPlacement(
                    AnyPlacement(), AnyPlacement()
                ),
            ),
        },
    )
    runner = build_compact_gurobi_solver_strategy(
        MisocpCompactFormulationConfig(0.5, kind),
        SOLVER_CONFIG,
        strategy_config=GurobiStrategyConfig(SolverStrategyKind.INCREMENTAL),
    )
    solution = await runner.solve(problem)
    assert_solution(problem, solution)
    diagnostics = runner.diagnostics
    assert len(diagnostics.attempts) == 2
    assert diagnostics.successful_insertions == 1
    assert all(a.termination == Termination.OPTIMAL for a in diagnostics.attempts)
    assert (
        next(f for f in solution.flow_solutions if f.flow_id == first)
        .components[0]
        .worker_id
        == slow
    )
    assert (
        next(f for f in solution.flow_solutions if f.flow_id == second)
        .components[0]
        .worker_id
        == fast
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", list(Kind))
@pytest.mark.parametrize("objective", list(ObjectiveKind))
async def test_native_strategy_covers_both_representations_and_all_objectives(
    kind, objective
):
    base = dag_problem(variants=1, workers=1, components=1, flows=2, priorities=2)
    problem = replace(base, policy=replace(base.policy, objective=objective))
    runner = build_compact_gurobi_solver_strategy(
        MisocpCompactFormulationConfig(0.5, kind),
        SOLVER_CONFIG,
        strategy_config=GurobiStrategyConfig("incremental"),
    )
    assert isinstance(runner, IncrementalFlowSolverStrategy)
    solution = await runner.solve(problem)
    assert_solution(problem, solution)
    assert len(runner.diagnostics.attempts) == 2
    assert all(
        attempt.termination == Termination.OPTIMAL
        for attempt in runner.diagnostics.attempts
    )
    assert solution.objective_value == pytest.approx(
        runner.diagnostics.attempts[-1].objective_value
    )


def test_default_strategy_and_config_validation():
    assert isinstance(
        build_compact_gurobi_solver_strategy(
            MisocpCompactFormulationConfig(0.5), SOLVER_CONFIG
        ),
        FullSolverStrategy,
    )
    for value in (0, -1, float("inf"), float("nan")):
        with pytest.raises(ValueError):
            GurobiStrategyConfig(insertion_time_limit_s=value)
    for value in (0, -1, 2.2, True):
        with pytest.raises(ValueError):
            GurobiSolverConfig(solution_limit=value)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", list(Kind))
async def test_single_flow_is_solved_once_and_retains_optimality(kind):
    problem = dag_problem(variants=1, workers=1, components=1)
    runner = build_compact_gurobi_solver_strategy(
        MisocpCompactFormulationConfig(0.5, kind),
        SOLVER_CONFIG,
        strategy_config=GurobiStrategyConfig("incremental"),
    )
    solution = await runner.solve(problem)
    assert {flow.flow_id for flow in solution.flow_solutions} == set(problem.flows)
    assert solution.flow_solutions[0].mean_response_time_s > 0
    assert len(runner.diagnostics.attempts) == 1
    assert runner.diagnostics.attempts[0].stage == "first_flow"
    assert runner.diagnostics.attempts[0].termination == Termination.OPTIMAL
    assert solution.is_optimal
