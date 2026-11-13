"""Each execution epigraph has its own candidate-gated full-model bound."""

import pytest
from test.optimizer.test_compact_exact_times import (
    _compiler,
    _fix_plan,
    _network_formulation,
    _solve,
)

from optimizer.application.ports.outbound.solver_result import SolverTerminationKind
from optimizer.domain.problem.compact.compact_optimization_problem import ObjectiveKind


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "first,second,termination",
    [
        (0.05, 0.05, SolverTerminationKind.OPTIMAL),
        (0.06, 0.06, SolverTerminationKind.OPTIMAL),
        (0.11, 0.0, SolverTerminationKind.INFEASIBLE),
    ],
)
async def test_component_bounds_do_not_share_an_execution_budget(
    first, second, termination
):
    data = _network_formulation(ObjectiveKind.CONSTANT)
    async with _compiler().compile_formulation(data) as compiled:
        flow, selected, source, _target = _fix_plan(
            compiled, selected_index=0, target_index=1, compute_component=2
        )
        times = compiled.variables.execution.execution_time_s
        # Each slot is bounded by 0.1 s; their sum may now exceed 0.1 s.
        compiled.native_model.addConstr(times[flow, selected, 1, source] == first)
        compiled.native_model.addConstr(times[flow, selected, 3, source] == second)
        result = await _solve(compiled)
        assert result.termination is termination


@pytest.mark.asyncio
@pytest.mark.parametrize("component", [1, 2, 3])
async def test_each_unselected_execution_time_is_zero(component):
    data = _network_formulation(ObjectiveKind.CONSTANT)
    async with _compiler().compile_formulation(data) as compiled:
        flow, _selected, source, _target = _fix_plan(
            compiled, selected_index=0, target_index=1, compute_component=2
        )
        excluded = data.configurations_by_flow[flow][1]
        variable = compiled.variables.execution.execution_time_s[
            flow, excluded, component, source
        ]
        compiled.native_model.addConstr(variable == 0.01)
        result = await _solve(compiled)
        assert result.termination is SolverTerminationKind.INFEASIBLE
