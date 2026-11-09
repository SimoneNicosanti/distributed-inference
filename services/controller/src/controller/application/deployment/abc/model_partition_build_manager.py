from abc import ABC, abstractmethod

from controller.domain.optimization.solving.optimization_solution import (
    OptimizationSolution,
)


class ModelPartitionBuildError(Exception):
    pass


class ModelPartitionBuildManager(ABC):
    @abstractmethod
    async def build_model_partitions_from_solution(
        self, solution: OptimizationSolution
    ) -> None: ...
