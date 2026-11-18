from dataclasses import asdict, dataclass
from enum import StrEnum

from optimizer.application.ports.outbound.solver_result import SolverTerminationKind


class IncrementalStage(StrEnum):
    FIRST_FLOW = "first_flow"
    INSERTION = "insertion"
    REPAIR = "repair"
    FINAL = "final"


@dataclass(frozen=True, slots=True)
class IncrementalAttemptDiagnostics:
    stage: IncrementalStage
    flow_count: int
    fixed_flow_count: int
    reason: str | None
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
    objective_before_final: float | None
    objective_after_final: float | None
    returned_objective_value: float | None
    returned_source: str | None
    wall_time_s: float

    @property
    def successful_restricted_insertions(self) -> int:
        return sum(
            attempt.stage == IncrementalStage.INSERTION and attempt.feasible_incumbent
            for attempt in self.attempts
        )

    @property
    def global_repair_count(self) -> int:
        return sum(
            attempt.stage == IncrementalStage.REPAIR for attempt in self.attempts
        )

    @property
    def incremental_construction_time_s(self) -> float:
        return sum(
            attempt.wall_time_s
            for attempt in self.attempts
            if attempt.stage
            in {IncrementalStage.FIRST_FLOW, IncrementalStage.INSERTION}
        )

    @property
    def global_repair_time_s(self) -> float:
        return sum(
            attempt.wall_time_s
            for attempt in self.attempts
            if attempt.stage == IncrementalStage.REPAIR
        )

    @property
    def final_global_time_s(self) -> float:
        return sum(
            attempt.wall_time_s
            for attempt in self.attempts
            if attempt.stage == IncrementalStage.FINAL
        )

    def as_record(self) -> dict[str, object]:
        return {
            **asdict(self),
            "successful_restricted_insertions": self.successful_restricted_insertions,
            "global_repair_count": self.global_repair_count,
            "incremental_construction_time_s": self.incremental_construction_time_s,
            "global_repair_time_s": self.global_repair_time_s,
            "final_global_time_s": self.final_global_time_s,
            "native_solve_count": sum(a.solver_called for a in self.attempts),
        }
