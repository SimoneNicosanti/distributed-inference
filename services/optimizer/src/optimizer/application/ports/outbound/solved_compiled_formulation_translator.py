from abc import ABC, abstractmethod

from optimizer.domain.solution.optimization_solution import (
    OptimizationSolution,
)


class SolvedCompiledFormulationTranslator[SolvedCompiledFormulationT](ABC):
    @abstractmethod
    async def translate_solved_formulation_model(
        self, solved_compiled_formulation: SolvedCompiledFormulationT
    ) -> OptimizationSolution: ...
