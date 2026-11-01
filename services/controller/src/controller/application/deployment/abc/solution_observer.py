from abc import ABC, abstractmethod

from controller.domain.optimization.solving.optimization_solution import (
    OptimizationSolution,
)


class SolutionObserver(ABC):
    @abstractmethod
    async def observe_new_solution(self) -> OptimizationSolution: ...
