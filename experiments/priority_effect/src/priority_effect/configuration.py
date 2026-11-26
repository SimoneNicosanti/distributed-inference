"""Experiment settings; reuse the scalability hardware/profile schemas."""

from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import Field, model_validator
from scalability_benchmark.configuration import (
    ExperimentConfig,
    LinkSettings,
    Settings,
    SolverSettings,
    TierSettings,
    load_configuration,
)

from optimizer.domain.problem.compact.compact_optimization_problem import ObjectiveKind


class Deadline(Settings):
    strategy: Literal["reference_response_multiplier"] = "reference_response_multiplier"
    multiplier: float = Field(default=3.0, ge=1)


class Workload(Settings):
    name: str
    category: Literal["homogeneous", "heterogeneous"]
    models: tuple[str, ...]
    # Optional exact assignment for each configured flow count.
    assignments: dict[int, tuple[str, ...]] = Field(default_factory=dict)
    distinct_models_per_flow: bool = True


class Optimization(Settings):
    objective: Literal[
        ObjectiveKind.ARRIVAL_WEIGHTED_RESPONSE_TIME,
        ObjectiveKind.ARRIVAL_WEIGHTED_NORMALIZED_RESPONSE_TIME,
        ObjectiveKind.ARRIVAL_WEIGHTED_NORMALIZED_DELAY,
    ] = ObjectiveKind.ARRIVAL_WEIGHTED_RESPONSE_TIME
    max_components: int = Field(default=5, ge=3)
    compute_utilization_limit: float = Field(default=0.9, gt=0, lt=1)
    network_utilization_limit: float = Field(default=0.8, gt=0, lt=1)
    relative_error: float = Field(default=0.1, gt=0)


type Strategy = Literal["full_p1", "full_p2"]


class PrioritySolverSettings(SolverSettings):
    measure_presolve_dimensions: bool = False


class Config(Settings):
    base_config: Path
    output: Path = Path("../results")
    seeds: tuple[int, ...] = (42,)
    flows: tuple[int, ...] = (2, 3, 5)
    target_loads: tuple[float, ...] = (0.3, 0.5, 0.7, 0.8)
    priority_counts: tuple[Literal[1, 2], ...] = (1, 2)
    strategies: tuple[Strategy, ...] = ("full_p1", "full_p2")
    workloads: tuple[Workload, ...]
    arrival_weight_range: tuple[float, float] = (0.8, 1.2)
    arrival_weights: dict[int, tuple[float, ...]] = Field(default_factory=dict)
    deadline: Deadline = Deadline()
    optimization: Optimization = Optimization()
    solver: PrioritySolverSettings
    hardware: dict[str, TierSettings]
    network: LinkSettings
    tight_gap: float = Field(default=0.01, ge=0)
    pilot_loads: tuple[float, ...] = (0.3, 0.8)
    reference_check: bool = True

    @model_validator(mode="after")
    def check(self) -> Self:
        for values in (self.seeds, self.flows, self.target_loads, self.pilot_loads):
            if not values or len(set(values)) != len(values):
                raise ValueError("Scenario dimensions must be nonempty and unique")
        if any(seed < 0 for seed in self.seeds) or any(n < 2 for n in self.flows):
            raise ValueError("Need nonnegative seeds and at least two flows")
        if self.priority_counts != (1, 2):
            raise ValueError("This experiment compares exactly priority_counts: [1, 2]")
        if not self.strategies or len(set(self.strategies)) != len(self.strategies):
            raise ValueError("Strategies must be nonempty and unique")
        if any(not 0 < load < 1 for load in (*self.target_loads, *self.pilot_loads)):
            raise ValueError("Loads must be in (0, 1)")
        if not set(self.pilot_loads) <= set(self.target_loads):
            raise ValueError("Pilot loads must belong to target_loads")
        if not 0 < self.arrival_weight_range[0] <= self.arrival_weight_range[1]:
            raise ValueError("Invalid arrival_weight_range")
        if set(self.hardware) != {"device", "edge"}:
            raise ValueError("Exactly one device and one edge hardware tier required")
        if (
            self.hardware["device"].backend != "cpu"
            or self.hardware["edge"].backend != "gpu"
        ):
            raise ValueError("Require CPU device and GPU edge")
        if self.solver.strategy != "full":
            raise ValueError("Paired study uses only the full solver strategy")
        if not self.workloads or len({w.name for w in self.workloads}) != len(
            self.workloads
        ):
            raise ValueError("Workload names must be nonempty and unique")
        for workload in self.workloads:
            if (
                not workload.name
                or not workload.models
                or len(set(workload.models)) != len(workload.models)
            ):
                raise ValueError("Need a workload name and unique model names")
            if workload.category == "homogeneous" and len(workload.models) != 1:
                raise ValueError("Homogeneous workloads require one model")
            if (
                workload.category == "heterogeneous"
                and workload.distinct_models_per_flow
                and len(workload.models) < max(self.flows)
            ):
                raise ValueError(
                    "Heterogeneous pool needs distinct models for every flow"
                )
            if (
                workload.category == "heterogeneous"
                and not workload.distinct_models_per_flow
                and any(n not in workload.assignments for n in self.flows)
            ):
                raise ValueError(
                    "Reused heterogeneous models require explicit assignments"
                )
            for n, names in workload.assignments.items():
                if (
                    n not in self.flows
                    or len(names) != n
                    or not set(names) <= set(workload.models)
                ):
                    raise ValueError(
                        "Explicit model assignment must match the flow count and pool"
                    )
                if (
                    workload.category == "heterogeneous"
                    and workload.distinct_models_per_flow
                    and len(set(names)) != n
                ):
                    raise ValueError(
                        "Heterogeneous flows need distinct model identities"
                    )
                if workload.category == "heterogeneous" and len(set(names)) < 2:
                    raise ValueError(
                        "Heterogeneous assignments require at least two models"
                    )
        for n, weights in self.arrival_weights.items():
            if n not in self.flows or len(weights) != n or any(w <= 0 for w in weights):
                raise ValueError(
                    "Arrival weights must be positive and match the flow count"
                )
        return self

    def benchmark_config(self) -> ExperimentConfig:
        """Adapt inputs for existing generators; never use their CPU calibration."""
        base = load_configuration(self.base_config)
        names = {name for workload in self.workloads for name in workload.models}
        if not names <= {model.name for model in base.catalog.models}:
            raise ValueError("Workload names must belong to the existing model catalog")
        return base.model_copy(
            update={
                "hardware": {**base.hardware, **self.hardware},
                "network": {**base.network, "device-edge": self.network},
                "catalog": base.catalog.model_copy(
                    update={
                        "models": tuple(
                            model
                            for model in base.catalog.models
                            if model.name in names
                        ),
                    }
                ),
                "optimization": base.optimization.model_copy(
                    update={
                        **self.optimization.model_dump(),
                        "priority_count": 1,
                    }
                ),
            }
        )


def load_config(path: Path) -> Config:
    path = path.resolve()
    config = Config.model_validate(yaml.safe_load(path.read_text()))
    return config.model_copy(
        update={
            "base_config": (path.parent / config.base_config).resolve(),
            "output": (path.parent / config.output).resolve(),
        }
    )
