from dataclasses import replace

import httpx
import pytest
from fastapi import FastAPI
from test.optimizer.test_full_optimizer_runtime import (
    assert_solution,
    compiler,
    pipeline,
)
from test.optimizer.test_single_configuration_contract import (
    controller_problem,
    single_configuration_problem,
)

from controller.adapters.outbound.optimizer.http_optimizer_gateway import (
    HttpOptimizerGateway,
)
from optimizer.adapters.inbound.http_optimizer_controller import build_optimizer_router
from optimizer.adapters.outbound.gurobi.entities.gurobi_solver_config import (
    GurobiSolverConfig,
)
from optimizer.adapters.outbound.gurobi.solver.gurobi_solver import GurobiSolver
from optimizer.application.formulation.compact.misocp.misocp_compact_formulation_builder import (
    MisocpCompactFormulationBuilder,
)
from optimizer.application.optimization_controller import DefaultOptimizerController
from optimizer.application.ports.outbound.solver_result import SolverTerminationKind
from optimizer.domain.formulation.compact.misocp.misocp_compact_formulation_config import (
    MisocpCompactFormulationConfig,
)
from optimizer.domain.problem.compact.compact_optimization_problem import ObjectiveKind


@pytest.mark.asyncio
@pytest.mark.parametrize("variant_count", [1, 2])
@pytest.mark.parametrize("objective", list(ObjectiveKind))
async def test_full_response_decomposition_with_one_configuration_per_variant(
    variant_count, objective
):
    problem = single_configuration_problem(variant_count)
    problem = replace(problem, policy=replace(problem.policy, objective=objective))
    assert len(problem.variants) == len(problem.configurations) == variant_count
    result = await pipeline().solve(problem)
    assert_solution(problem, result)


@pytest.mark.asyncio
async def test_an_excluded_variant_must_not_contribute_execution_costs():
    problem = single_configuration_problem(2)
    problem = replace(
        problem,
        policy=replace(
            problem.policy, objective=ObjectiveKind.ARRIVAL_WEIGHTED_ACCURACY
        ),
    )
    formulation = MisocpCompactFormulationBuilder().build_formulation(
        problem, MisocpCompactFormulationConfig(0.5)
    )
    flow, worker = formulation.flows[0], formulation.workers[0]
    selected, excluded = formulation.configurations_by_flow[flow]
    assert selected.model_variant_id != excluded.model_variant_id
    async with compiler().compile_formulation(formulation) as model:
        for key, value in ((selected, 1), (excluded, 0)):
            variable = model.variables.assignment.configuration_selected[flow, key]
            variable.LB = variable.UB = value
        model.variables.execution.execution_time_s[flow, excluded, 1, worker].LB = 0.1
        result = await GurobiSolver().solve_compiled_formulation(
            model, GurobiSolverConfig(time_limit_s=10, threads=1)
        )
        assert result.termination is SolverTerminationKind.INFEASIBLE


@pytest.mark.asyncio
async def test_single_configuration_contract_through_http_and_native_full_optimizer():
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
    assert result.flow_solutions[0].input_shapes == {"features": (1, 3), "mask": (1,)}
