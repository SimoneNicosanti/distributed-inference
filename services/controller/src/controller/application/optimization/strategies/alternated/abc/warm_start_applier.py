from abc import ABC, abstractmethod

from controller.domain.optimization.solving.optimization_solution import (
    OptimizationSolution,
)


class WarmStartApplier[FormulationModelT](ABC):
    @abstractmethod
    async def apply_warm_start(
        self,
        formulation_model: FormulationModelT,
        warm_start_solution: OptimizationSolution,
    ) -> None: ...
