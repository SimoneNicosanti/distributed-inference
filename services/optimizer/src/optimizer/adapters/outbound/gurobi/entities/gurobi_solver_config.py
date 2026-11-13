import math
from collections.abc import Callable
from dataclasses import dataclass, field

from optimizer.adapters.outbound.gurobi.entities.gurobi_run_statistics import (
    GurobiRunStatistics,
)


@dataclass(frozen=True, slots=True)
class GurobiSolverConfig:
    output_flag: bool = False
    time_limit_s: float | None = None
    mip_gap: float | None = None
    threads: int | None = None
    seed: int | None = None
    observer: Callable[[GurobiRunStatistics], None] | None = field(
        default=None, repr=False, compare=False
    )
    record_incumbent_history: bool = False

    def __post_init__(self) -> None:
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
