from dataclasses import asdict, dataclass
from enum import StrEnum

from optimizer.application.ports.outbound.solver_result import SolverTerminationKind


class IncrementalStage(StrEnum):
    FIRST_FLOW = "first_flow"
    INSERTION = "insertion"


@dataclass(frozen=True, slots=True)
class IncrementalAttemptDiagnostics:
    stage: IncrementalStage
    flow_count: int
    termination: SolverTerminationKind | None
    feasible_incumbent: bool
    objective_value: float | None
    construction_time_s: float
    solver_runtime_s: float | None
    time_to_first_incumbent_s: float | None
    wall_time_s: float
    error_type: str | None = None
    error_details: str | None = None
    solver_called: bool = False


@dataclass(frozen=True, slots=True)
class IncrementalFlowDiagnostics:
    attempts: tuple[IncrementalAttemptDiagnostics, ...]
    completed_flow_count: int
    returned_objective_value: float | None
    returned_source: str | None
    wall_time_s: float

    @property
    def successful_insertions(self) -> int:
        return sum(
            attempt.stage == IncrementalStage.INSERTION and attempt.feasible_incumbent
            for attempt in self.attempts
        )

    @property
    def prefix_wall_time_s(self) -> float:
        return sum(attempt.wall_time_s for attempt in self.attempts)

    def as_record(self) -> dict[str, object]:
        return {
            **asdict(self),
            "successful_insertions": self.successful_insertions,
            "prefix_wall_time_s": self.prefix_wall_time_s,
            "native_solve_count": sum(a.solver_called for a in self.attempts),
        }
