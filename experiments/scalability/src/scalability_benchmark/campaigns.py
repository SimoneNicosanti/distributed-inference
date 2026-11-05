"""Deterministic campaign enumeration, independent of profiling and solving."""

from collections.abc import Iterator
from dataclasses import asdict, dataclass, replace
from typing import Literal

from controller.domain.optimization.solving.solver_config import ObjectiveKind
from scalability_benchmark.configuration import ExperimentConfig

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

    @property
    def worker_count(self) -> int:
        return 1 + self.edge_count + self.cloud_count

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
    for spec in _instance_specs(config, name):
        for objective in selected:
            yield replace(spec, objective=objective)


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
