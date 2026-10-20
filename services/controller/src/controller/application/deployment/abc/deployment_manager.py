from abc import ABC, abstractmethod

from controller.domain.optimization.optimization_solution import OptimizationSolution


class DeploymentManager(ABC):
    @abstractmethod
    async def deploy_solution(self, solution: OptimizationSolution) -> None: ...
