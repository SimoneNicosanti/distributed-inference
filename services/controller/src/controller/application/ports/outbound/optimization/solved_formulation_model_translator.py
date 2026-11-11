from abc import ABC, abstractmethod

from controller.domain.optimization.solving.optimization_solution import (
    OptimizationSolution,
)


class SolvedFormulationModelTranslator[SolvedFormulationModelT](ABC):
    @abstractmethod
    async def translate_solved_formulation_model(
        self, solved_formulation_model: SolvedFormulationModelT
    ) -> OptimizationSolution: ...
