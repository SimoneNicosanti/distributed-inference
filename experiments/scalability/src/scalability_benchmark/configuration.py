"""Validated experiment assumptions. Paths resolve relative to YAML root."""

from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from controller.domain.optimization.solving.solver_config import ObjectiveKind


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class ModelSettings(Settings):
    name: str
    filename: str
    input_shape: tuple[int, int, int, int]
    accuracy_rank: int = Field(ge=0)

    @model_validator(mode="after")
    def check(self) -> Self:
        if self.input_shape[0] != 1 or any(d <= 0 for d in self.input_shape):
            raise ValueError(f"{self.name}: expected positive batch-1 input shape")
        if Path(self.filename).name != self.filename or not self.filename.endswith(
            ".onnx"
        ):
            raise ValueError("Model filename must be a local .onnx basename")
        return self


class CatalogSettings(Settings):
    directory: Path
    selection: Literal["nested_permutation"]
    dataset: str
    accuracy_range: tuple[float, float]
    models: tuple[ModelSettings, ...]

    @model_validator(mode="after")
    def check(self) -> Self:
        if not self.models:
            raise ValueError("The model pool must not be empty")
        for values in (
            [m.name for m in self.models],
            [m.filename for m in self.models],
            [m.accuracy_rank for m in self.models],
        ):
            if len(set(values)) != len(self.models):
                raise ValueError(
                    "Model names, filenames and accuracy ranks must be unique"
                )
        if (
            not self.dataset
            or not 0 <= self.accuracy_range[0] < self.accuracy_range[1] <= 1
        ):
            raise ValueError("Accuracy requires a dataset and a range within [0, 1]")
        return self


class ExportSettings(Settings):
    seed: int = Field(ge=0)
    opset: int = Field(ge=13)
    weights: Literal["random", "local", "download"]
    optimization: Literal["basic"]


class TierSettings(Settings):
    backend: Literal["cpu", "gpu"]
    cpu_cores: int = Field(gt=0)
    ram_gib: float = Field(gt=0)
    vram_gib: float = Field(ge=0)
    compute_flops_s: float = Field(gt=0)
    memory_bytes_s: float = Field(gt=0)
    overhead_s: float = Field(ge=0)

    @model_validator(mode="after")
    def check(self) -> Self:
        if (self.backend == "gpu") != (self.vram_gib > 0):
            raise ValueError("GPU tiers require VRAM; CPU tiers must have no VRAM")
        return self


class Efficiency(Settings):
    operators: tuple[str, ...]
    compute: float = Field(gt=0, le=1)
    memory: float = Field(gt=0, le=1)


class ExecutionSettings(Settings):
    jitter_fraction: float = Field(ge=0, lt=1)
    service_scv: Literal[0]
    memory_strategy: Literal["weights_plus_peak_live_diagnostic"]
    groups: dict[str, Efficiency]
    fallback: Efficiency


class LinkSettings(Settings):
    one_way_ms: tuple[float, float]
    throughput_mbps: tuple[float, float]

    @model_validator(mode="after")
    def check(self) -> Self:
        if not 0 <= self.one_way_ms[0] <= self.one_way_ms[1]:
            raise ValueError("Invalid latency range")
        if not 0 < self.throughput_mbps[0] <= self.throughput_mbps[1]:
            raise ValueError("Invalid throughput range")
        return self


class FlowSettings(Settings):
    pool_size: int = Field(gt=0)
    arrival_process: Literal["poisson"]
    minimum_accuracy: Literal[0]
    deadline_s: tuple[float, float]
    arrival_weight_range: tuple[float, float]
    reference_model: str
    reference_tier: Literal["device"]
    calibration: Literal["full_pool_reference_utilization"]
    max_energy_consumption: float = Field(gt=0)

    @model_validator(mode="after")
    def check(self) -> Self:
        for name, bounds in (
            ("deadline_s", self.deadline_s),
            ("arrival_weight_range", self.arrival_weight_range),
        ):
            if not 0 < bounds[0] <= bounds[1]:
                raise ValueError(f"Invalid {name} range")
        return self


class OptimizationSettings(Settings):
    max_components: int = Field(gt=0)
    priority_count: int = Field(gt=0)
    compute_utilization_limit: float = Field(gt=0, lt=1)
    network_utilization_limit: float = Field(gt=0, lt=1)
    relative_error: float = Field(gt=0)
    objective: ObjectiveKind


class SolverSettings(Settings):
    formulation: Literal["compact"] = "compact"
    timeout_s: float = Field(gt=0)
    relative_gap: float = Field(ge=0)
    threads: int | None = Field(default=None, ge=0)
    seed: int = Field(ge=0, le=2_000_000_000)
    output_flag: bool


class CampaignSettings(Settings):
    enabled: bool
    loads: tuple[float, ...]
    flows: tuple[int, ...]
    models: tuple[int, ...]


class PilotCase(Settings):
    flows: int = Field(ge=1)
    models: int = Field(ge=1)
    infrastructure_index: int = Field(ge=0, le=10)


class PilotSettings(Settings):
    timeout_s: float = Field(gt=0)
    load: float = Field(gt=0, lt=1)
    cases: tuple[PilotCase, ...]


class ExperimentConfig(Settings):
    root: Path
    seeds: tuple[int, ...]
    catalog: CatalogSettings
    export: ExportSettings
    hardware: dict[str, TierSettings]
    infrastructure: tuple[tuple[int, int], ...]
    execution: ExecutionSettings
    network: dict[str, LinkSettings]
    flows: FlowSettings
    optimization: OptimizationSettings
    solver: SolverSettings
    main: CampaignSettings
    sensitivity: CampaignSettings
    pilot: PilotSettings
    output: Path

    @model_validator(mode="after")
    def check(self) -> Self:
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("Seeds must be nonempty and unique")
        if any(seed < 0 for seed in self.seeds):
            raise ValueError("Instance seeds cannot be negative")
        if set(self.hardware) != {"device", "edge", "cloud"}:
            raise ValueError(
                "Exactly device, edge and cloud hardware tiers are required"
            )
        if self.hardware["device"].backend != "cpu":
            raise ValueError("The device must be CPU-only")
        if any(self.hardware[tier].backend != "gpu" for tier in ("edge", "cloud")):
            raise ValueError("Edge and cloud must provide GPUs")
        progression = tuple((i, 0) for i in range(11))
        if self.infrastructure != progression:
            raise ValueError(
                "Infrastructure must preserve all 11 ordered configurations"
            )
        if set(self.network) != {
            "device-edge",
            "edge-edge",
            "edge-cloud",
            "device-cloud",
            "cloud-cloud",
        }:
            raise ValueError("All five network link classes are required")
        if self.flows.reference_model not in {m.name for m in self.catalog.models}:
            raise ValueError("Calibration reference model is outside the model pool")
        for campaign in (self.main, self.sensitivity):
            for values, maximum, name in (
                (campaign.flows, self.flows.pool_size, "flows"),
                (campaign.models, len(self.catalog.models), "models"),
            ):
                if not values or tuple(sorted(set(values))) != values:
                    raise ValueError(
                        f"Campaign {name} must be nonempty, unique and sorted"
                    )
                if any(not 1 <= value <= maximum for value in values):
                    raise ValueError(f"Campaign {name} outside supported pool")
            if (
                not campaign.loads
                or tuple(sorted(set(campaign.loads))) != campaign.loads
            ):
                raise ValueError("Campaign loads must be nonempty, unique and sorted")
            if any(not 0 < load < 1 for load in campaign.loads):
                raise ValueError("Campaign loads must be in (0, 1)")
        if not self.pilot.cases or len(set(self.pilot.cases)) != len(self.pilot.cases):
            raise ValueError("Pilot cases must be nonempty and unique")
        if any(
            case.flows > self.flows.pool_size or case.models > len(self.catalog.models)
            for case in self.pilot.cases
        ):
            raise ValueError("Pilot flows or models outside supported pool")
        operators = [
            op for group in self.execution.groups.values() for op in group.operators
        ]
        if len(set(operators)) != len(operators):
            raise ValueError("Execution operator groups must not overlap")
        return self

    @property
    def model_directory(self) -> Path:
        return (self.root / self.catalog.directory).resolve()

    def assumptions(self) -> dict[str, object]:
        return self.model_dump(mode="json", exclude={"root", "output"})


def load_configuration(path: Path) -> ExperimentConfig:
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    config = ExperimentConfig.model_validate(raw)
    return config.model_copy(
        update={"root": (path.resolve().parent / config.root).resolve()}
    )
