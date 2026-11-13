from typing import override

from optimizer.application.abc.solver_strategy import SolverStrategy
from optimizer.application.ports.inbound.optimizer_controller import OptimizerController
from optimizer.domain.problem.compact.compact_optimization_problem import (
    CompactOptimizationProblem,
)
from optimizer.domain.problem.optimization_problem import OptimizationProblem
from optimizer.domain.solution.optimization_solution import OptimizationSolution


class DefaultOptimizerController(OptimizerController):
    def __init__(
        self, compact_strategy: SolverStrategy[CompactOptimizationProblem]
    ) -> None:
        super().__init__()
        self._strategy = compact_strategy

    @override
    async def optimize(self, problem: OptimizationProblem) -> OptimizationSolution:

        if isinstance(problem, CompactOptimizationProblem):
            return await self._strategy.solve(problem)

        raise TypeError()
