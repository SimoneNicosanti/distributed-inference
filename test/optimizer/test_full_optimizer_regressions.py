"""Expected invariants exposing defects found during the static review."""

from contextlib import asynccontextmanager
from dataclasses import replace

import pytest
from test.optimizer.test_full_optimizer_runtime import (
    RecordingSolver,
    compiler,
    pipeline,
)
from test.optimizer.test_misocp_compact_formulation_builder import _build, _problem

from optimizer.adapters.outbound.gurobi.entities.gurobi_solver_config import (
    GurobiSolverConfig,
)
from optimizer.adapters.outbound.gurobi.solver.gurobi_solver import GurobiSolver
from optimizer.application.abc.solver_strategy import SolveError
from optimizer.application.ports.outbound.solver_result import SolverTerminationKind
from optimizer.domain.problem.compact.compact_optimization_problem import ObjectiveKind


@pytest.mark.asyncio
async def test_unselected_configuration_must_not_have_positive_execution_time():
    problem = _problem()
    problem = replace(
        problem,
        policy=replace(
            problem.policy, objective=ObjectiveKind.ARRIVAL_WEIGHTED_ACCURACY
        ),
    )
    formulation = _build(problem)
    flow = formulation.flows[0]
    worker = formulation.workers[0]
    selected, excluded = formulation.configurations_by_flow[flow]
    async with compiler().compile_formulation(formulation) as model:
        for key, value in ((selected, 1), (excluded, 0)):
            variable = model.variables.assignment.configuration_selected[flow, key]
            variable.LB = variable.UB = value
        # Ask whether a forbidden auxiliary cost is feasible under the model.
        model.variables.execution.execution_time_s[flow, excluded, 1, worker].LB = 0.1
        result = await GurobiSolver().solve_compiled_formulation(
            model, GurobiSolverConfig(time_limit_s=10, threads=1)
        )
        assert result.termination is SolverTerminationKind.INFEASIBLE, (
            "Unselected configuration admitted execution cost 0.1"
        )


@pytest.mark.asyncio
async def test_aggregate_response_must_equal_translated_component_times():
    problem = _problem()
    problem = replace(
        problem,
        policy=replace(
            problem.policy, objective=ObjectiveKind.ARRIVAL_WEIGHTED_ACCURACY
        ),
    )

    class WitnessCompiler:
        @asynccontextmanager
        async def compile_formulation(self, formulation):
            flow, worker = formulation.flows[0], formulation.workers[0]
            selected, excluded = formulation.configurations_by_flow[flow]
            async with compiler().compile_formulation(formulation) as model:
                for key, value in ((selected, 1), (excluded, 0)):
                    variable = model.variables.assignment.configuration_selected[
                        flow, key
                    ]
                    variable.LB = variable.UB = value
                model.variables.execution.execution_time_s[
                    flow, excluded, 1, worker
                ].LB = 0.1
                yield model

    result = await pipeline(model_compiler=WitnessCompiler()).solve(problem)
    flow = result.flow_solutions[0]
    decomposed = flow.mean_batch_fill_time_s + sum(
        component.mean_execution_time_s
        + component.mean_execution_queue_time_s
        + component.mean_transmission_time_s
        + component.transmission_latency_s
        for component in flow.components
    )
    assert flow.mean_response_time_s == pytest.approx(decomposed, abs=1e-6)


@pytest.mark.asyncio
async def test_full_error_preserves_native_infeasible_termination():
    problem = _problem()
    worker = next(iter(problem.workers))
    problem = replace(
        problem,
        workers={
            worker: replace(problem.workers[worker], available_inference_memory_bytes=1)
        },
    )
    solver = RecordingSolver()
    with pytest.raises(SolveError) as caught:
        await pipeline(solver=solver).solve(problem)
    assert solver.output.termination is SolverTerminationKind.INFEASIBLE
    assert getattr(caught.value, "termination", None) is solver.output.termination
