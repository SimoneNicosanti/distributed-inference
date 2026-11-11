import math
from dataclasses import dataclass

from controller.domain.optimization.solving.formulation.compact.compact_objective_kind import (
    CompactObjectiveKind,
)


@dataclass(frozen=True, slots=True)
class CompactFormulationConfig:
    max_components: int
    priority_count: int
    compute_utilization_limit: float
    network_utilization_limit: float
    relative_error: float
    objective: CompactObjectiveKind

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
