from abc import ABC, abstractmethod

from optimizer.domain.solution.optimization_solution import OptimizationSolution


class IncrementalModelAdapter[CompiledFormulationT, SolvedCompiledFormulationT](ABC):
    """Backend operations for partial MIP starts and incumbent validation."""

    @abstractmethod
    async def initialize_model(
        self,
        model: CompiledFormulationT,
        solution: OptimizationSolution,
    ) -> None: ...

    @abstractmethod
    def is_valid_solution(self, model: SolvedCompiledFormulationT) -> bool: ...
