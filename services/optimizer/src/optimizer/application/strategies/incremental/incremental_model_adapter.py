from abc import ABC, abstractmethod

from optimizer.domain.solution.optimization_solution import OptimizationSolution


class IncrementalModelAdapter[CompiledFormulationT, SolvedCompiledFormulationT](ABC):
    """Backend operations; fixing decisions must never fix performance auxiliaries."""

    @abstractmethod
    async def initialize_model(
        self,
        model: CompiledFormulationT,
        solution: OptimizationSolution,
        *,
        fix_decisions: bool,
    ) -> None: ...

    @abstractmethod
    def is_valid_solution(self, model: SolvedCompiledFormulationT) -> bool: ...
