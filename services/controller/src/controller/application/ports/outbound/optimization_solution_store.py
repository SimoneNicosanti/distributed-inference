from abc import ABC, abstractmethod

from controller.domain.optimization.optimization_solution import OptimizationSolution


class OptimizationSolutionStore(ABC):
    @abstractmethod
    def put_solution(self, solution: OptimizationSolution) -> None: ...

    @abstractmethod
    def get_solution(self) -> OptimizationSolution: ...
