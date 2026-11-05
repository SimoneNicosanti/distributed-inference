import math
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum, auto


class ObjectiveKind(StrEnum):
    ARRIVAL_WEIGHTED_RESPONSE_TIME = auto()
    ARRIVAL_WEIGHTED_ACCURACY = auto()
    DEADLINE_VIOLATIONS = auto()
    ARRIVAL_WEIGHTED_NORMALIZED_DELAY = auto()


class FormulationKind(StrEnum):
    COMPACT = "compact"


@dataclass(frozen=True, slots=True)
class ProblemConfig:
    max_components: int
    priority_count: int
    compute_utilization_limit: float
    network_utilization_limit: float
    relative_error: float

    def __post_init__(self) -> None:
        if self.max_components < 1:
            raise ValueError("max_components must be positive")
        if self.priority_count < 1:
            raise ValueError("priority_count must be positive")
        for name, value in (
            ("compute_utilization_limit", self.compute_utilization_limit),
            ("network_utilization_limit", self.network_utilization_limit),
        ):
            if not 0 < value < 1:
                raise ValueError(f"{name} must be strictly between zero and one")
        if not math.isfinite(self.relative_error) or self.relative_error <= 0:
            raise ValueError("relative_error must be finite and positive")


@dataclass(frozen=True, slots=True)
class SolverRunStatistics:
    """Native solver observations, collected before its model is disposed."""

    status: int | None
    runtime_s: float | None
    solution_count: int
    objective_value: float | None
    objective_bound: float | None
    relative_gap: float | None
    construction_time_s: float
    time_limit_s: float
    mip_gap_target: float
    threads: int
    seed: int
    formulation: str = "compact"
    continuous_variables: int = 0
    binary_variables: int = 0
    integer_variables: int = 0
    linear_constraints: int = 0
    quadratic_constraints: int = 0
    general_constraints: int = 0
    node_count: float | None = None
    objective: str = ObjectiveKind.ARRIVAL_WEIGHTED_RESPONSE_TIME.value


@dataclass(frozen=True, slots=True)
class SolverConfig:
    objective: ObjectiveKind
    output_flag: bool = False
    time_limit_s: float | None = None
    mip_gap: float | None = None
    threads: int | None = None
    seed: int | None = None
    observer: Callable[[SolverRunStatistics], None] | None = field(
        default=None, repr=False, compare=False
    )
    formulation: FormulationKind = FormulationKind.COMPACT

    def __post_init__(self) -> None:
        object.__setattr__(self, "objective", ObjectiveKind(self.objective))
        object.__setattr__(self, "formulation", FormulationKind(self.formulation))
        if self.time_limit_s is not None and (
            not math.isfinite(self.time_limit_s) or self.time_limit_s <= 0
        ):
            raise ValueError("time_limit_s must be positive")
        if self.mip_gap is not None and (
            not math.isfinite(self.mip_gap) or self.mip_gap < 0
        ):
            raise ValueError("mip_gap cannot be negative")
        if self.threads is not None and self.threads < 0:
            raise ValueError("threads cannot be negative")
        if self.seed is not None and not 0 <= self.seed <= 2_000_000_000:
            raise ValueError("seed must be between zero and 2000000000")
