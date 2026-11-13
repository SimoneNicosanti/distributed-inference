"""Reduced epigraphs allow slack but preserve selection gating and budgets."""

from dataclasses import replace

import pytest
from test.optimizer.test_compact_exact_times import (
    _compiler,
    _fix_plan,
    _network_formulation,
    _solve,
)
from test.optimizer.test_full_optimizer_runtime import assert_solution, compiler

from optimizer.adapters.outbound.gurobi.result_translation.formulations.compact.gurobi_misocp_compact_solved_compiled_formulation_translator import (
    GurobiMisocpCompactSolvedCompiledFormulationTranslator,
)
from optimizer.application.formulation.compact.misocp.misocp_compact_formulation_builder import (
    MisocpCompactFormulationBuilder,
)
from optimizer.application.ports.outbound.solver_result import SolverTerminationKind
from optimizer.domain.formulation.compact.misocp.misocp_compact_formulation_config import (
    MisocpCompactFormulationConfig,
)
from optimizer.domain.problem.compact.compact_optimization_problem import ObjectiveKind


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "first,second,termination",
    [
        (0.05, 0.05, SolverTerminationKind.OPTIMAL),
        (0.06, 0.06, SolverTerminationKind.INFEASIBLE),
    ],
)
async def test_execution_budget_is_shared_by_all_component_slots(
    first, second, termination
):
    data = _network_formulation(ObjectiveKind.CONSTANT)
    async with _compiler().compile_formulation(data) as compiled:
        flow, selected, source, _target = _fix_plan(
            compiled, selected_index=0, target_index=1, compute_component=2
        )
        times = compiled.variables.execution.execution_time_s
        # Neither component executes the computational layer on source. The
        # reduced formulation permits slack, including in an inactive slot,
        # but the two values share the source's total model budget of 0.1 s.
        compiled.native_model.addConstr(times[flow, selected, 1, source] == first)
        compiled.native_model.addConstr(times[flow, selected, 3, source] == second)
        result = await _solve(compiled)
        assert result.termination is termination


@pytest.mark.asyncio
@pytest.mark.parametrize("component", [1, 2, 3])
async def test_unselected_candidate_has_no_execution_budget(component):
    data = _network_formulation(ObjectiveKind.CONSTANT)
    async with _compiler().compile_formulation(data) as compiled:
        flow, _selected, source, _target = _fix_plan(
            compiled, selected_index=0, target_index=1, compute_component=2
        )
        excluded = data.configurations_by_flow[flow][1]
        time = compiled.variables.execution.execution_time_s[
            flow, excluded, component, source
        ]
        compiled.native_model.addConstr(time == 0.01)
        result = await _solve(compiled)
        assert result.termination is SolverTerminationKind.INFEASIBLE


@pytest.mark.asyncio
async def test_execution_can_have_slack_on_an_unplaced_worker():
    data = _network_formulation(ObjectiveKind.CONSTANT)
    async with _compiler().compile_formulation(data) as compiled:
        flow, selected, source, _target = _fix_plan(
            compiled, selected_index=0, target_index=1, compute_component=2
        )
        time = compiled.variables.execution.execution_time_s[flow, selected, 2, source]
        compiled.native_model.addConstr(time == 0.05)
        result = await _solve(compiled)
        assert result.termination is SolverTerminationKind.OPTIMAL


@pytest.mark.asyncio
async def test_execution_cannot_fall_below_the_selected_layers_cost():
    data = _network_formulation(ObjectiveKind.CONSTANT)
    async with _compiler().compile_formulation(data) as compiled:
        flow, selected, _source, target = _fix_plan(
            compiled, selected_index=0, target_index=1, compute_component=2
        )
        time = compiled.variables.execution.execution_time_s[flow, selected, 2, target]
        compiled.native_model.addConstr(time == 0.15)  # Actual execution is 0.2 s.
        result = await _solve(compiled)
        assert result.termination is SolverTerminationKind.INFEASIBLE


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case,termination",
    [
        ("above_true", SolverTerminationKind.OPTIMAL),
        ("above_slowest_link_bound", SolverTerminationKind.INFEASIBLE),
        ("below_true", SolverTerminationKind.INFEASIBLE),
        ("unselected", SolverTerminationKind.INFEASIBLE),
        ("unplaced", SolverTerminationKind.INFEASIBLE),
        ("empty_cut", SolverTerminationKind.INFEASIBLE),
        ("no_outgoing_links", SolverTerminationKind.INFEASIBLE),
    ],
)
async def test_reduced_transmission_bounds(case, termination):
    data = _network_formulation(ObjectiveKind.CONSTANT)
    async with _compiler().compile_formulation(data) as compiled:
        flow, selected, source, target = _fix_plan(
            compiled, selected_index=0, target_index=1, compute_component=2
        )
        excluded = data.configurations_by_flow[flow][1]
        ghost, isolated = data.problem.flows[flow].workers[3:]
        times = compiled.variables.network.transmission_time_s
        # The cut carries 1 MB: selected-link time 0.1 s, slowest-link bound 1 s.
        variable, value = {
            "above_true": (times[flow, selected, 1, source], 0.2),
            "above_slowest_link_bound": (times[flow, selected, 1, source], 1.1),
            "below_true": (times[flow, selected, 1, source], 0.05),
            "unselected": (times[flow, excluded, 1, source], 0.05),
            "unplaced": (times[flow, selected, 1, ghost], 0.05),
            "empty_cut": (times[flow, selected, 2, target], 0.05),
            "no_outgoing_links": (times[flow, selected, 1, isolated], 0.05),
        }[case]
        compiled.native_model.addConstr(variable == value)
        result = await _solve(compiled)
        assert result.termination is termination


@pytest.mark.asyncio
@pytest.mark.parametrize("objective", list(ObjectiveKind))
async def test_full_translation_recomputes_execution_despite_forced_slack(objective):
    problem = _network_formulation(objective).problem
    flow = next(iter(problem.flows))
    workers = problem.flows[flow].workers[:2]
    problem = replace(
        problem,
        workers={worker: problem.workers[worker] for worker in workers},
        flows={flow: replace(problem.flows[flow], workers=workers)},
        execution_times_s={
            key: times
            for key, times in problem.execution_times_s.items()
            if key[1] in workers
        },
        connections=tuple(
            link
            for link in problem.connections
            if link.source in workers and link.target in workers
        ),
        policy=replace(problem.policy, max_components=2),
    )
    data = MisocpCompactFormulationBuilder().build_formulation(
        problem, MisocpCompactFormulationConfig(0.5)
    )
    async with compiler().compile_formulation(data) as compiled:
        flow, selected, source, _target = _fix_plan(
            compiled, selected_index=0, target_index=1, compute_component=2
        )
        # Source executes only the zero-cost input layer. The auxiliary value
        # is deliberately inflated within the cumulative budget.
        time = compiled.variables.execution.execution_time_s[flow, selected, 1, source]
        compiled.native_model.addConstr(time == 0.05)
        result = await _solve(compiled)
        assert result.termination is SolverTerminationKind.OPTIMAL
        solution = await GurobiMisocpCompactSolvedCompiledFormulationTranslator().translate_solved_formulation_model(
            compiled
        )
        assert_solution(problem, solution)
        translated = solution.flow_solutions[0]
        assert translated.components[0].mean_execution_time_s == pytest.approx(0)
        assert translated.components[1].mean_execution_time_s == pytest.approx(0.2)
        modeled_execution = sum(
            var.X for var in compiled.variables.execution.execution_time_s.values()
        )
        actual_execution = sum(
            component.mean_execution_time_s for component in translated.components
        )
        assert modeled_execution - actual_execution >= 0.05 - 1e-7
        assert translated.mean_response_time_s == pytest.approx(
            compiled.variables.qos.response_time_s[flow].X
            - modeled_execution
            + actual_execution
        )
        assert solution.objective_value == compiled.native_model.ObjVal
