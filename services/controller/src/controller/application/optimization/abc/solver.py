from abc import ABC, abstractmethod

from controller.domain.optimization.optimization_context import OptimizationContext
from controller.domain.optimization.optimization_request import (
    OptimizationRequestsByFlow,
)
from controller.domain.optimization.optimization_solution import OptimizationSolution
from controller.domain.optimization.solver_config import ProblemConfig


class Solver(ABC):
    @abstractmethod
    async def solve(
        self,
        requests_by_flow: OptimizationRequestsByFlow,
        context: OptimizationContext,
        problem_config: ProblemConfig,
    ) -> OptimizationSolution: ...
