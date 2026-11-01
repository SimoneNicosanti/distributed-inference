from abc import ABC, abstractmethod

from controller.domain.optimization.solving.optimization_solution import (
    OptimizationSolution,
)


class ActuationManager(ABC):
    @abstractmethod
    async def actuate_solution(self, solution: OptimizationSolution) -> None: ...
