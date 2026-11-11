from abc import ABC, abstractmethod

from controller.domain.optimization.request.optimization_request import (
    OptimizationRequestsByFlow,
)
from controller.domain.optimization.solving.optimization_context import (
    OptimizationContext,
)
from controller.domain.optimization.solving.optimization_solution import (
    OptimizationSolution,
)


class SolveError(Exception):
    pass


class SolverStrategy(ABC):
    @abstractmethod
    async def solve(
        self,
        requests_by_flow: OptimizationRequestsByFlow,
        context: OptimizationContext,
    ) -> OptimizationSolution: ...
