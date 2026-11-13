from abc import ABC, abstractmethod
from contextlib import AbstractAsyncContextManager

from optimizer.application.strategies.alternated.abc.phase_generator import (
    PhaseInfo,
)
from optimizer.domain.solution.optimization_solution import (
    OptimizationSolution,
)


class PhaseConstraintsApplier[FormulationModelT](ABC):
    @abstractmethod
    def apply_next_phase_constraints(
        self,
        formulation_model: FormulationModelT,
        incumbent: OptimizationSolution,
        phase_info: PhaseInfo,
    ) -> AbstractAsyncContextManager[None]: ...
