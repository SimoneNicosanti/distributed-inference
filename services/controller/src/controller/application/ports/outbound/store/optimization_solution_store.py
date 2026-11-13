from abc import ABC, abstractmethod

from controller.domain.optimization.solution.enriched_optimization_solution import (
    EnrichedOptimizationSolution,
)


class OptimizationSolutionStore(ABC):
    @abstractmethod
    async def put_solution(self, solution: EnrichedOptimizationSolution) -> None: ...

    @abstractmethod
    async def get_current_solution(self) -> EnrichedOptimizationSolution | None: ...
