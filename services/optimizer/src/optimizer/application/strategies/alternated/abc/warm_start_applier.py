from abc import ABC, abstractmethod

from optimizer.domain.solution.optimization_solution import (
    OptimizationSolution,
)


class WarmStartApplier[FormulationModelT](ABC):
    @abstractmethod
    async def apply_warm_start(
        self,
        formulation_model: FormulationModelT,
        warm_start_solution: OptimizationSolution,
    ) -> None: ...
