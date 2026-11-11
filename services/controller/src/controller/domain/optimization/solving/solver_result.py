from dataclasses import dataclass
from enum import StrEnum, auto


class SolverTerminationKind(StrEnum):
    """Why a solve stopped; solution availability is reported separately."""

    OPTIMAL = auto()
    INFEASIBLE = auto()
    UNBOUNDED = auto()
    INFEASIBLE_OR_UNBOUNDED = auto()
    TIME_LIMIT = auto()
    ITERATION_LIMIT = auto()
    NODE_LIMIT = auto()
    SOLUTION_LIMIT = auto()
    OBJECTIVE_LIMIT = auto()
    WORK_LIMIT = auto()
    MEMORY_LIMIT = auto()
    INTERRUPTED = auto()
    NUMERICAL_FAILURE = auto()
    SUBOPTIMAL = auto()
    UNKNOWN = auto()


@dataclass(frozen=True, slots=True)
class SolverOutput[SolvedFormulationModelT, SolverStatisticsT]:
    termination: SolverTerminationKind
    solved_model: SolvedFormulationModelT | None
    objective_value: float | None
    objective_bound: float | None
    runtime_s: float | None
    statistics: SolverStatisticsT
