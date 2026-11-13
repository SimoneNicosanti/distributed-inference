from abc import ABC, abstractmethod

from controller.domain.optimization.context.optimization_context import (
    OptimizationContext,
)
from controller.domain.optimization.problem.optimization_problem import (
    OptimizationProblem,
)
from controller.domain.optimization.request.optimization_request import (
    OptimizationRequestsByFlow,
)


class OptimizationProblemFactory(ABC):
    @abstractmethod
    async def build_optimization_problem(
        self, requests_by_flow: OptimizationRequestsByFlow, context: OptimizationContext
    ) -> OptimizationProblem: ...
