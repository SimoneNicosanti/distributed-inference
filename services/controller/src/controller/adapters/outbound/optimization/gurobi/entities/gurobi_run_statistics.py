from dataclasses import dataclass

from controller.adapters.outbound.optimization.gurobi.entities.gurobi_incumbent_sample import (
    GurobiIncumbentSample,
)


@dataclass(frozen=True, slots=True)
class GurobiRunStatistics:
    """Native solver observations, collected before its model is disposed."""

    native_status: int | None
    solution_count: int
    relative_gap: float | None
    construction_time_s: float
    time_limit_s: float
    mip_gap_target: float
    threads: int
    seed: int
    continuous_variables: int = 0
    binary_variables: int = 0
    integer_variables: int = 0
    linear_constraints: int = 0
    quadratic_constraints: int = 0
    general_constraints: int = 0
    node_count: float | None = None
    time_to_first_incumbent_s: float | None = None
    incumbent_history: tuple[GurobiIncumbentSample, ...] = ()
