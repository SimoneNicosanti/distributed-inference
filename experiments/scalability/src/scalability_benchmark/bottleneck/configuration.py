"""Isolated study settings and deterministic one-factor-at-a-time planning."""

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import Field, model_validator

from optimizer.domain.formulation.compact.misocp.compact_formulation_kind import (
    CompactFormulationKind,
)
from optimizer.domain.problem.compact.compact_optimization_problem import ObjectiveKind
from scalability_benchmark.campaigns import RunSpec
from scalability_benchmark.configuration import ExperimentConfig, Settings
from scalability_benchmark.randomness import fingerprint

# Complete existing device + 0..10 edge progression; never rewrite the main catalog.
INFRASTRUCTURES = tuple((edge, 0) for edge in range(11))


class SolverSettings(Settings):
    time_limit_seconds: float = Field(default=300, gt=0)
    mip_gap: float = Field(default=0.01, ge=0)
    threads: int = Field(default=1, ge=0)
    seed: int = Field(default=0, ge=0, le=2_000_000_000)
    console: bool = False
    measure_presolve: bool = True
    presolve_probe_seconds: float = Field(default=30, gt=0)


class ReferenceSettings(Settings):
    validation_seconds: float = Field(default=30, gt=0)


class Study(Settings):
    base_config: Path = Path("../../config/performance.yaml")
    output: Path = Path("../results/default-b-reference")
    flow_counts: tuple[int, ...] = (5, 10, 20, 30, 50, 75, 100)
    seeds: tuple[int, ...] = (42, 43)
    loads: tuple[float, ...] = (0.8,)
    objectives: tuple[ObjectiveKind, ...] = (
        ObjectiveKind.ARRIVAL_WEIGHTED_NORMALIZED_RESPONSE_TIME,
    )
    formulation: CompactFormulationKind = CompactFormulationKind.CUMULATIVE
    infrastructure_indices: tuple[int, ...] = (1,)
    model_count: int = Field(default=3, gt=0)
    max_components: int = Field(default=5, gt=0)
    priority_count: int = Field(default=2, gt=0)
    start_modes: tuple[Literal["cold", "warm"], ...] = ("cold", "warm")
    include_primary_fixed: bool = False
    anchor_flow_count: int = Field(default=20, gt=0)
    candidate_model_counts: tuple[int, ...] = ()
    component_counts: tuple[int, ...] = ()
    worker_counts: tuple[int, ...] = ()
    priority_counts: tuple[int, ...] = ()
    deadline_s: tuple[float, float] | None = None
    solver: SolverSettings = SolverSettings()
    reference: ReferenceSettings = ReferenceSettings()

    @model_validator(mode="after")
    def check(self) -> Self:
        for name in (
            "flow_counts",
            "seeds",
            "loads",
            "objectives",
            "infrastructure_indices",
            "start_modes",
        ):
            values = getattr(self, name)
            if not values or len(values) != len(set(values)):
                raise ValueError(f"{name} must be nonempty and unique")
        for name in (
            "flow_counts",
            "candidate_model_counts",
            "component_counts",
            "worker_counts",
            "priority_counts",
        ):
            values = getattr(self, name)
            if tuple(sorted(set(values))) != values or any(v < 1 for v in values):
                raise ValueError(f"{name} must be positive, sorted and unique")
        if any(s < 0 for s in self.seeds) or any(not 0 < x < 1 for x in self.loads):
            raise ValueError("Invalid seed or load")
        if any(not 0 <= i < 11 for i in self.infrastructure_indices) or any(
            w > 11 for w in self.worker_counts
        ):
            raise ValueError("Infrastructure supports 1..11 total workers")
        if (
            self.deadline_s is not None
            and not 0 < self.deadline_s[0] <= self.deadline_s[1]
        ):
            raise ValueError("Invalid deadline range")
        return self

    @property
    def ablations(self) -> tuple[str, ...]:
        return (
            ("A", "B", "C", "D", "P", "E")
            if self.include_primary_fixed
            else ("A", "B", "C", "D", "E")
        )


@dataclass(frozen=True)
class Case:
    spec: RunSpec
    max_components: int
    priority_count: int
    sweep: str = "flows"
    sweep_value: int = 0

    def record(self) -> dict[str, Any]:
        return {
            **self.spec.as_record(),
            "objective": self.spec.objective.value,
            "formulation": self.spec.formulation.value,
            "max_components": self.max_components,
            "priority_count": self.priority_count,
            "sweep": self.sweep,
            "sweep_value": self.sweep_value,
        }


def cases(study: Study) -> tuple[Case, ...]:
    result: list[Case] = []
    for seed in study.seeds:
        for load in study.loads:
            for objective in study.objectives:
                for index in study.infrastructure_indices:
                    edge, cloud = INFRASTRUCTURES[index]
                    base = Case(
                        RunSpec(
                            "main",
                            seed,
                            load,
                            study.anchor_flow_count,
                            study.model_count,
                            edge,
                            cloud,
                            study.solver.time_limit_seconds,
                            objective,
                            study.formulation,
                        ),
                        study.max_components,
                        study.priority_count,
                    )
                    result.extend(
                        replace(
                            base, spec=replace(base.spec, flow_count=n), sweep_value=n
                        )
                        for n in study.flow_counts
                    )
                    for axis, values in (
                        ("models", study.candidate_model_counts),
                        ("components", study.component_counts),
                        ("workers", study.worker_counts),
                        ("priorities", study.priority_counts),
                    ):
                        for value in values:
                            item = replace(base, sweep=axis, sweep_value=value)
                            if axis == "models":
                                item = replace(
                                    item, spec=replace(item.spec, model_count=value)
                                )
                            elif axis == "workers":
                                item = replace(
                                    item,
                                    spec=replace(
                                        item.spec, edge_count=value - 1, cloud_count=0
                                    ),
                                )
                            elif axis == "components":
                                item = replace(item, max_components=value)
                            else:
                                item = replace(item, priority_count=value)
                            result.append(item)
    # Multiple selected infrastructure families may share an identical worker-sweep point.
    return tuple({fingerprint(asdict(item)): item for item in result}.values())


def load_study(path: Path) -> tuple[Study, ExperimentConfig]:
    raw = yaml.safe_load(path.read_text())
    study = Study.model_validate(raw)
    directory = path.resolve().parent
    base_path = (directory / study.base_config).resolve()
    base_raw = yaml.safe_load(base_path.read_text())
    count = max((*study.flow_counts, study.anchor_flow_count))
    # These unrelated campaigns are not run. Validate a small, consistent envelope.
    base_raw["flows"]["pool_size"] = count
    if study.deadline_s is not None:
        base_raw["flows"]["deadline_s"] = study.deadline_s
    envelope = {
        "enabled": False,
        "loads": list(study.loads),
        "flows": [1],
        "models": [1],
    }
    base_raw["main"] = envelope
    base_raw["sensitivity"] = envelope
    base_raw["pilot"] = {
        "timeout_s": 1,
        "load": study.loads[0],
        "cases": [{"flows": 1, "models": 1, "infrastructure_index": 0}],
    }
    base = ExperimentConfig.model_validate(base_raw)
    maximum_models = max((study.model_count, *study.candidate_model_counts))
    if maximum_models > len(base.catalog.models):
        raise ValueError("Requested candidate count exceeds the original model catalog")
    # The main validator currently accepts only its four active entries. The diagnostic
    # view restores all eleven supported prefixes without changing that validator/file.
    base = base.model_copy(
        update={
            "root": (base_path.parent / base.root).resolve(),
            "infrastructure": INFRASTRUCTURES,
        }
    )
    study = study.model_copy(
        update={
            "base_config": base_path,
            "output": (directory / study.output).resolve(),
        }
    )
    return study, base
