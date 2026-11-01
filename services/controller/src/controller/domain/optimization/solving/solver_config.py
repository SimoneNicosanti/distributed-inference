import math
from dataclasses import dataclass
from enum import StrEnum, auto


class ObjectiveKind(StrEnum):
    ARRIVAL_WEIGHTED_RESPONSE_SLACK = auto()
    ARRIVAL_WEIGHTED_NORMALIZED_RESPONSE_SLACK = auto()


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
class SolverConfig:
    objective: ObjectiveKind
    output_flag: bool = False
    time_limit_s: float | None = None
    mip_gap: float | None = None

    def __post_init__(self) -> None:
        if self.time_limit_s is not None and self.time_limit_s <= 0:
            raise ValueError("time_limit_s must be positive")
        if self.mip_gap is not None and self.mip_gap < 0:
            raise ValueError("mip_gap cannot be negative")
