"""Normalized deadline consumption stays sensitive below hard deadlines."""

from collections.abc import Awaitable, Callable
from dataclasses import replace
from typing import cast
from uuid import UUID

import pytest
from test.optimizer.test_single_configuration_contract import (
    controller_problem,
    single_configuration_problem,
)

from controller.adapters.outbound.optimizer.mappers.optimization_problem_mapper import (
    OptimizationProblemMapper as ControllerMapper,
)
from controller.domain.optimization.problem.compact_optimization_problem import (
    CompactOptimizationProblem as ControllerProblem,
)
from controller.domain.optimization.problem.compact_optimization_problem import (
    ObjectiveKind as ControllerObjectiveKind,
)
from integration.dto.optimizer.optimization_problem_dto import ObjectiveKindDto
from integration.optimizer.http.optimizer_http_schema import OptimizeRequest
from optimizer.adapters.inbound.mappers.optimization_problem_mapper import (
    OptimizationProblemMapper,
)
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
from optimizer.application.formulation.compact.misocp.misocp_compact_formulation_builder import (
    MisocpCompactFormulationBuilder,
)
from optimizer.application.ports.outbound.solver_result import SolverTerminationKind
from optimizer.domain.formulation.compact.misocp.compact_formulation_kind import (
    CompactFormulationKind,
)
from optimizer.domain.formulation.compact.misocp.misocp_compact_formulation import (
    MisocpCompactFormulation,
)
from optimizer.domain.formulation.compact.misocp.misocp_compact_formulation_config import (
    MisocpCompactFormulationConfig,
)
from optimizer.domain.problem.compact.compact_optimization_problem import (
    CompactOptimizationProblem,
    ObjectiveKind,
)
from optimizer.domain.solution.optimization_solution import OptimizationSolution
from shared.flow.flow import FlowId

NORMALIZED_RESPONSE = ObjectiveKind.ARRIVAL_WEIGHTED_NORMALIZED_RESPONSE_TIME


def _problem() -> CompactOptimizationProblem:
    factory = cast(
        Callable[[], CompactOptimizationProblem], single_configuration_problem
    )
    problem = factory()
    return replace(
        problem, policy=replace(problem.policy, objective=NORMALIZED_RESPONSE)
    )


async def _solve(
    problem: CompactOptimizationProblem,
    representation: CompactFormulationKind,
    priorities: dict[FlowId, int] | None = None,
) -> tuple[SolverTerminationKind, OptimizationSolution | None]:
    formulation = MisocpCompactFormulationBuilder().build_formulation(
        problem, MisocpCompactFormulationConfig(0.5, representation)
    )
    compiler = GurobiFormulationCompiler(
        GurobiMisocpCompactFormulationModelBuilder(), non_convex=0
    )
    async with compiler.compile_formulation(formulation) as model:
        if priorities is not None:
            worker = next(iter(problem.workers))
            for flow, priority in priorities.items():
                variable = model.variables.compute_queue.component_priority[
                    flow, 1, worker, priority
                ]
                variable.LB = variable.UB = 1
        output = await GurobiSolver[
            MisocpCompactFormulation, CompactVariables
        ]().solve_compiled_formulation(
            model, GurobiSolverConfig(time_limit_s=10, threads=1, seed=0, mip_gap=0)
        )
        solution = (
            await GurobiMisocpCompactSolvedCompiledFormulationTranslator().translate_solved_formulation_model(
                output.solved_model
            )
            if output.solved_model is not None
            else None
        )
        return output.termination, solution


@pytest.mark.asyncio
@pytest.mark.parametrize("representation", list(CompactFormulationKind))
async def test_objective_is_request_weighted_fraction_of_deadline(
    representation: CompactFormulationKind,
) -> None:
    problem = _problem()
    first = next(iter(problem.flows))
    second = FlowId(flow_id=UUID(int=5))
    problem = replace(
        problem,
        flows={
            first: replace(
                problem.flows[first], arrival_rate_per_s=0.5, max_response_time_s=1.0
            ),
            second: replace(
                problem.flows[first], arrival_rate_per_s=1.5, max_response_time_s=4.0
            ),
        },
    )
    termination, solution = await _solve(problem, representation)
    assert termination is SolverTerminationKind.OPTIMAL
    assert solution is not None
    total_rate = sum(flow.arrival_rate_per_s for flow in problem.flows.values())
    expected = (
        sum(
            problem.flows[flow.flow_id].arrival_rate_per_s
            * flow.mean_response_time_s
            / problem.flows[flow.flow_id].max_response_time_s
            for flow in solution.flow_solutions
        )
        / total_rate
    )
    assert solution.objective_value == pytest.approx(expected, abs=1e-6)
    assert 0 < solution.objective_value < 1
    assert all(
        flow.mean_response_time_s < problem.flows[flow.flow_id].max_response_time_s
        for flow in solution.flow_solutions
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("representation", list(CompactFormulationKind))
async def test_normalized_response_preserves_hard_deadlines(
    representation: CompactFormulationKind,
) -> None:
    problem = _problem()
    flow = next(iter(problem.flows))
    problem = replace(
        problem, flows={flow: replace(problem.flows[flow], max_response_time_s=0.01)}
    )
    termination, solution = await _solve(problem, representation)
    assert termination is SolverTerminationKind.INFEASIBLE
    assert solution is None


@pytest.mark.asyncio
@pytest.mark.parametrize("representation", list(CompactFormulationKind))
async def test_priorities_improve_objective_even_when_all_deadlines_are_satisfied(
    representation: CompactFormulationKind,
) -> None:
    problem = _problem()
    first = next(iter(problem.flows))
    second = FlowId(flow_id=UUID(int=5))
    problem = replace(
        problem,
        flows={
            first: replace(problem.flows[first], max_response_time_s=1.0),
            second: replace(problem.flows[first], max_response_time_s=10.0),
        },
        policy=replace(problem.policy, priority_count=2),
    )
    _, preferred = await _solve(problem, representation, {first: 1, second: 2})
    _, reversed_order = await _solve(problem, representation, {first: 2, second: 1})
    _, optimized = await _solve(problem, representation)
    assert (
        preferred is not None and reversed_order is not None and optimized is not None
    )
    for solution in (preferred, reversed_order, optimized):
        assert solution.is_optimal
        assert all(
            flow.mean_response_time_s < problem.flows[flow.flow_id].max_response_time_s
            for flow in solution.flow_solutions
        )
    assert preferred.objective_value < reversed_order.objective_value - 1e-6
    assert optimized.objective_value <= preferred.objective_value + 1e-6


@pytest.mark.asyncio
@pytest.mark.parametrize("objective", list(ObjectiveKind))
async def test_objective_contract_survives_controller_json_and_optimizer_mapping(
    objective: ObjectiveKind,
) -> None:
    assert (
        {kind.value for kind in ObjectiveKind}
        == {kind.value for kind in ControllerObjectiveKind}
        == {kind.value for kind in ObjectiveKindDto}
    )
    factory = cast(Callable[[], Awaitable[ControllerProblem]], controller_problem)
    original = await factory()
    original = replace(
        original,
        policy=replace(
            original.policy, objective=ControllerObjectiveKind(objective.value)
        ),
    )
    request = OptimizeRequest(problem=ControllerMapper.to_dto(original))
    decoded = OptimizeRequest.model_validate_json(request.model_dump_json())
    mapped = OptimizationProblemMapper.from_dto(decoded.problem)
    assert mapped.policy.objective is objective
