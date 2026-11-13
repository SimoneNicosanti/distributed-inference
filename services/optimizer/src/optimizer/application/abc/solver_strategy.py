from abc import ABC, abstractmethod

from optimizer.domain.problem.optimization_problem import OptimizationProblem
from optimizer.domain.solution.optimization_solution import (
    OptimizationSolution,
)


class SolveError(Exception):
    pass


class SolverStrategy[OptimizationProblemT: OptimizationProblem](ABC):
    @abstractmethod
    async def solve(self, problem: OptimizationProblemT) -> OptimizationSolution: ...
