from abc import ABC, abstractmethod

from controller.domain.optimization.solving.enriched_optimization_solution import (
    EnrichedOptimizationSolution,
)


class ActuationManager(ABC):
    @abstractmethod
    async def actuate_solution(
        self, solution: EnrichedOptimizationSolution
    ) -> None: ...
