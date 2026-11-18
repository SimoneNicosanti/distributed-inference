"""Deterministic campaign enumeration, independent of profiling and solving."""

from collections.abc import Iterator
from dataclasses import asdict, dataclass, replace
from typing import Literal

from optimizer.domain.formulation.compact.misocp.compact_formulation_kind import (
    CompactFormulationKind,
)
from optimizer.domain.problem.compact.compact_optimization_problem import (
    ObjectiveKind,
)
from scalability_benchmark.configuration import BENCHMARK_OBJECTIVES, ExperimentConfig

type CampaignName = Literal["pilot", "main", "sensitivity"]


@dataclass(frozen=True)
class RunSpec:
    campaign: CampaignName
    seed: int
    load: float
    flow_count: int
    model_count: int
    edge_count: int
    cloud_count: int
    timeout_s: float
    objective: ObjectiveKind = ObjectiveKind.ARRIVAL_WEIGHTED_RESPONSE_TIME
    formulation: CompactFormulationKind = CompactFormulationKind.ASSIGNMENT
    strategy: Literal["full", "incremental"] = "full"
    insertion_time_limit_s: float | None = None
    repair_time_limit_s: float | None = None
    final_time_limit_s: float | None = None

    @property
    def worker_count(self) -> int:
        return 1 + self.edge_count + self.cloud_count

    @property
    def solver_budget_upper_bound_s(self) -> float:
        if self.strategy == "full":
            return self.timeout_s
        return (
            self.flow_count * (self.insertion_time_limit_s or self.timeout_s)
            + (self.flow_count - 1) * (self.repair_time_limit_s or self.timeout_s)
            + (self.final_time_limit_s or self.timeout_s)
        )

    def as_record(self) -> dict[str, object]:
        return {**asdict(self), "worker_count": self.worker_count}


def campaign_specs(
    config: ExperimentConfig,
    name: CampaignName,
    *,
    objectives: tuple[ObjectiveKind, ...] | None = None,
) -> Iterator[RunSpec]:
    selected = tuple(
        dict.fromkeys(
            ObjectiveKind(value)
            for value in (
                objectives
                if objectives is not None
                else (config.optimization.objective,)
            )
        )
    )
    if not selected:
        raise ValueError("Select at least one objective")
    if any(objective not in BENCHMARK_OBJECTIVES for objective in selected):
        raise ValueError("Select one of the four benchmark objectives")
    for spec in _instance_specs(config, name):
        for objective in selected:
            yield replace(
                spec,
                objective=objective,
                formulation=config.solver.formulation,
                strategy=config.solver.strategy,
                insertion_time_limit_s=config.solver.insertion_time_limit_s,
                repair_time_limit_s=config.solver.repair_time_limit_s,
                final_time_limit_s=config.solver.final_time_limit_s,
            )


def _instance_specs(config: ExperimentConfig, name: CampaignName) -> Iterator[RunSpec]:
    for seed in sorted(config.seeds):
        if name == "pilot":
            for case in config.pilot.cases:
                edge, cloud = config.infrastructure[case.infrastructure_index]
                yield RunSpec(
                    name,
                    seed,
                    config.pilot.load,
                    case.flows,
                    case.models,
                    edge,
                    cloud,
                    config.pilot.timeout_s,
                )
        else:
            campaign = config.main if name == "main" else config.sensitivity
            # Explicit sensitivity invocation opts in even when disabled by default.
            for load in campaign.loads:
                for flows in campaign.flows:
                    for models in campaign.models:
                        for edge, cloud in config.infrastructure:
                            yield RunSpec(
                                name,
                                seed,
                                load,
                                flows,
                                models,
                                edge,
                                cloud,
                                config.solver.timeout_s,
                            )
