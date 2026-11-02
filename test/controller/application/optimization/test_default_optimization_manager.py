from typing import cast
from unittest.mock import AsyncMock, Mock, call

import pytest

from controller.application.optimization.abc.optimization_context_builder import (
    OptimizationContextBuilder,
)
from controller.application.optimization.abc.optimization_solution_enricher import (
    OptimizationSolutionEnricher,
)
from controller.application.optimization.abc.solver import Solver
from controller.application.optimization.default_optimization_manager import (
    DefaultOptimizationManager,
)
from controller.application.ports.outbound.store.optimization_solution_store import (
    OptimizationSolutionStore,
)
from controller.domain.optimization.request.optimization_request import (
    OptimizationRequest,
    OptimizationRequestsByFlow,
)
from controller.domain.optimization.solving.enriched_optimization_solution import (
    EnrichedOptimizationSolution,
)
from controller.domain.optimization.solving.optimization_context import (
    OptimizationContext,
)
from controller.domain.optimization.solving.optimization_solution import (
    OptimizationSolution,
)
from controller.domain.optimization.solving.solver_config import ProblemConfig
from shared.flow.flow import FlowId


@pytest.mark.asyncio
@pytest.mark.parametrize("enrichment_fails", [False, True])
async def test_builds_context_then_solves_enriches_and_stores_solution(
    enrichment_fails: bool,
) -> None:
    flow_id = cast(FlowId, Mock(spec=FlowId))
    requests_by_flow: OptimizationRequestsByFlow = {
        flow_id: cast(OptimizationRequest, Mock(spec=OptimizationRequest))
    }
    context = Mock(spec=OptimizationContext)
    context.static_profiles_by_variant = {}
    solution = OptimizationSolution((), 0, True)
    enriched_solution = EnrichedOptimizationSolution(solution, {})
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
    enricher = Mock(spec=OptimizationSolutionEnricher)
    enricher.enrich_solution.return_value = enriched_solution
    if enrichment_fails:
        enricher.enrich_solution.side_effect = ValueError("enrichment failed")
    solution_store = Mock(spec=OptimizationSolutionStore)
    sequence = Mock()
    sequence.attach_mock(context_builder.build_optimization_context, "build_context")
    sequence.attach_mock(solver.solve, "solve")
    sequence.attach_mock(enricher.enrich_solution, "enrich")
    sequence.attach_mock(solution_store.put_solution, "store")

    manager = DefaultOptimizationManager(
        context_builder,
        solver,
        enricher,
        solution_store,
        problem_config,
    )

    if enrichment_fails:
        with pytest.raises(ValueError, match="enrichment failed"):
            await manager.optimize_inference_distribution(requests_by_flow)
        solution_store.put_solution.assert_not_awaited()
    else:
        await manager.optimize_inference_distribution(requests_by_flow)
        solution_store.put_solution.assert_awaited_once_with(enriched_solution)

    context_builder.build_optimization_context.assert_awaited_once_with()
    solver.solve.assert_awaited_once_with(
        requests_by_flow,
        context,
        problem_config,
    )
    enricher.enrich_solution.assert_called_once_with(
        solution, context.static_profiles_by_variant
    )
    expected_calls = [
        call.build_context(),
        call.solve(requests_by_flow, context, problem_config),
        call.enrich(solution, context.static_profiles_by_variant),
    ]
    if not enrichment_fails:
        expected_calls.append(call.store(enriched_solution))
    assert sequence.mock_calls == expected_calls
