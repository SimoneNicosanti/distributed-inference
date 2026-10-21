from typing import cast
from unittest.mock import AsyncMock, Mock

import pytest

from controller.application.optimization.abc.optimization_context_builder import (
    OptimizationContextBuilder,
)
from controller.application.optimization.abc.solver import Solver
from controller.application.optimization.default_optimization_manager import (
    DefaultOptimizationManager,
)
from controller.application.ports.outbound.store.optimization_solution_store import (
    OptimizationSolutionStore,
)
from controller.domain.optimization.optimization_context import OptimizationContext
from controller.domain.optimization.optimization_request import (
    OptimizationRequest,
    OptimizationRequestsByFlow,
)
from controller.domain.optimization.optimization_solution import OptimizationSolution
from controller.domain.optimization.solver_config import ProblemConfig
from shared.flow.flow import FlowId


@pytest.mark.asyncio
async def test_builds_context_then_solves_and_stores_solution() -> None:
    flow_id = cast(FlowId, Mock(spec=FlowId))
    requests_by_flow: OptimizationRequestsByFlow = {
        flow_id: cast(OptimizationRequest, Mock(spec=OptimizationRequest))
    }
    context = Mock(spec=OptimizationContext)
    solution = Mock(spec=OptimizationSolution)
    problem_config = ProblemConfig(
        max_components=3,
        priority_count=2,
        compute_utilization_limit=0.9,
        network_utilization_limit=0.8,
        relative_error=0.1,
    )

    context_builder = Mock(spec=OptimizationContextBuilder)
    context_builder.build_optimization_context = AsyncMock(return_value=context)
    solver = Mock(spec=Solver)
    solver.solve = AsyncMock(return_value=solution)
    solution_store = Mock(spec=OptimizationSolutionStore)

    manager = DefaultOptimizationManager(
        context_builder,
        solver,
        solution_store,
        problem_config,
    )

    await manager.optimize_inference_distribution(requests_by_flow)

    context_builder.build_optimization_context.assert_awaited_once_with()
    solver.solve.assert_awaited_once_with(
        requests_by_flow,
        context,
        problem_config,
    )
    solution_store.put_solution.assert_called_once_with(solution)
