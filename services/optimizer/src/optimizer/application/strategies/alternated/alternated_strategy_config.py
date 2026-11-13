from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AlternatedStrategyConfig:
    num_optimization_steps: int
