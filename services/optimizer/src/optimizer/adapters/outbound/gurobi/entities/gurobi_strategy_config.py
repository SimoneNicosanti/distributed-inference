import math
from dataclasses import dataclass

from optimizer.application.strategies.solver_strategy_kind import SolverStrategyKind


@dataclass(frozen=True, slots=True)
class GurobiStrategyConfig:
    strategy: SolverStrategyKind = SolverStrategyKind.FULL
    insertion_time_limit_s: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "strategy", SolverStrategyKind(self.strategy))
        value = self.insertion_time_limit_s
        if value is not None and (not math.isfinite(value) or value <= 0):
            raise ValueError("Strategy stage time limits must be finite and positive")
