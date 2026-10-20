from typing import override

from controller.application.optimization.abc.optimization_context_builder import (
    OptimizationContextBuilder,
)
from controller.application.optimization.abc.solver import Solver
from controller.application.ports.inbound.optimization.optimization_manager import (
    OptimizationManager,
)
from controller.application.ports.outbound.optimization_solution_store import (
    OptimizationSolutionStore,
)
from controller.domain.optimization.optimization_request import (
    OptimizationRequestsByFlow,
)
from controller.domain.optimization.solver_config import ProblemConfig


class DefaultOptimizationManager(OptimizationManager):
    def __init__(
        self,
        optimization_context_builder: OptimizationContextBuilder,
        solver: Solver,
        solution_store: OptimizationSolutionStore,
        problem_config: ProblemConfig,
    ) -> None:
        self._optimization_context_builder = optimization_context_builder
        self._solver = solver
        self._solution_store = solution_store
        self._problem_config = problem_config

    @override
    async def optimize_inference_distribution(
        self,
        requests_by_flow: OptimizationRequestsByFlow,
    ) -> None:
        context = await self._optimization_context_builder.build_optimization_context()
        solution = await self._solver.solve(
            requests_by_flow,
            context,
            self._problem_config,
        )
        self._solution_store.put_solution(solution)
