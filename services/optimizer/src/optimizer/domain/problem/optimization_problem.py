from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class OptimizationProblem:
    """Base for optimizer-owned problem representations received by its input port."""
