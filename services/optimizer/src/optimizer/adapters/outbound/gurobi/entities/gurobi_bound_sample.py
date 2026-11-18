from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GurobiBoundSample:
    runtime_s: float
    objective_bound: float
    node_count: float
