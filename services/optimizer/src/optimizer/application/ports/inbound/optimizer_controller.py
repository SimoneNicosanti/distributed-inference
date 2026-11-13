from abc import ABC, abstractmethod

from optimizer.domain.problem.optimization_problem import OptimizationProblem
from optimizer.domain.solution.optimization_solution import OptimizationSolution


class OptimizerController(ABC):
    @abstractmethod
    async def optimize(self, problem: OptimizationProblem) -> OptimizationSolution: ...
