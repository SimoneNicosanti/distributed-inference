from abc import ABC, abstractmethod

from controller.domain.optimization.solving.enriched_optimization_solution import (
    EnrichedOptimizationSolution,
)


class SolutionObserver(ABC):
    @abstractmethod
    async def observe_new_solution(self) -> EnrichedOptimizationSolution: ...
