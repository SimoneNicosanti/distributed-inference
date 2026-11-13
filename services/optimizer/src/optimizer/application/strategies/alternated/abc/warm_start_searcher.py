from abc import ABC, abstractmethod

from optimizer.domain.problem.optimization_problem import OptimizationProblem
from optimizer.domain.solution.optimization_solution import OptimizationSolution


class WarmStartSearcher[OptimizationProblemT: OptimizationProblem](ABC):
    @abstractmethod
    async def search_warm_start(
        self, problem: OptimizationProblemT
    ) -> OptimizationSolution: ...
