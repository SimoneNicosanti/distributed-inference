from abc import ABC, abstractmethod

from controller.domain.optimization.problem.optimization_problem import (
    OptimizationProblem,
)
from controller.domain.optimization.solution.optimization_solution import (
    OptimizationSolution,
)


class OptimizerGateway(ABC):
    @abstractmethod
    async def optimize(self, problem: OptimizationProblem) -> OptimizationSolution: ...
