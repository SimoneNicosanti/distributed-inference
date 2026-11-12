from abc import ABC, abstractmethod
from contextlib import AbstractAsyncContextManager

from controller.application.optimization.strategies.alternated.abc.phase_generator import (
    PhaseInfo,
)
from controller.domain.optimization.solving.optimization_solution import (
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
