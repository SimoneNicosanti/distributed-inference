from abc import ABC, abstractmethod

from controller.domain.optimization.optimization_solution import OptimizationSolution


class OptimizationSolutionStore(ABC):
    @abstractmethod
    async def put_solution(self, solution: OptimizationSolution) -> None: ...

    @abstractmethod
    async def get_current_solution(self) -> OptimizationSolution | None: ...
