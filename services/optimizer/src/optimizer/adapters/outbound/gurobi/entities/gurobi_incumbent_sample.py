from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GurobiIncumbentSample:
    runtime_s: float
    objective_value: float | None
