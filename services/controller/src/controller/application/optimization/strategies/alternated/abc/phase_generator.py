from abc import ABC, abstractmethod
from dataclasses import dataclass

from controller.domain.optimization.solving.optimization_solution import (
    OptimizationSolution,
)
from controller.domain.optimization.solving.solver_result import SolverTerminationKind


@dataclass(frozen=True, slots=True)
class PhaseInfo:
    """Phase identifier; ends marks completion of an alternation cycle."""

    name: str
    ends: bool


@dataclass(frozen=True, slots=True)
class PhaseOutcome:
    """Observed phase result, without retaining the mutable solver model.

    Improvement is measured against the incumbent before the phase and is
    positive for a better candidate, for both minimization and maximization.
    None denotes an unavailable measurement, rather than a zero improvement.
    """

    phase: PhaseInfo
    termination: SolverTerminationKind
    has_candidate: bool
    accepted: bool = False
    candidate_objective_value: float | None = None
    improvement: float | None = None
    runtime_s: float | None = None


@dataclass(frozen=True, slots=True)
class PhaseGenerationContext:
    """Per-optimization snapshot supplied by the strategy to the phase policy.

    History contains completed alternating phases, excluding warm-start solves.
    The strategy owns budgets and the tolerance used to count stagnant cycles.
    A None remaining budget means that no such limit is configured.
    """

    incumbent: OptimizationSolution
    history: tuple[PhaseOutcome, ...] = ()
    cycles_without_improvement: int = 0
    remaining_steps: int | None = None
    remaining_time_s: float | None = None

    @property
    def last_outcome(self) -> PhaseOutcome | None:
        return self.history[-1] if self.history else None

    @property
    def completed_steps(self) -> int:
        return len(self.history)

    @property
    def completed_cycles(self) -> int:
        return sum(outcome.phase.ends for outcome in self.history)


class PhaseGenerator(ABC):
    @abstractmethod
    def next_phase(self, context: PhaseGenerationContext) -> PhaseInfo: ...
