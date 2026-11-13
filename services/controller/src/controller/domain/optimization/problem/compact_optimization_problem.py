import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum, auto
from types import MappingProxyType

from controller.domain.backend import Backend
from controller.domain.optimization.problem.optimization_problem import (
    OptimizationProblem,
)
from controller.domain.optimization.request.flow_endpoint_placement import (
    FlowEndpointPlacement,
)
from controller.domain.optimization.request.local_worker_config import LocalWorkerConfig
from controller.domain.profile.model_variant_static_profile import (
    ModelInputConcreteShapes,
)
from controller.domain.profile.model_variant_topology import EdgeDescription
from controller.domain.profile.network_topology import NetworkConnection
from shared.flow.flow import FlowId
from shared.model.keys import LayerKey, TensorKey
from shared.model.model_variant import ModelVariantId, PerformanceMetric
from shared.service.service import WorkerId


class ObjectiveKind(StrEnum):
    ARRIVAL_WEIGHTED_RESPONSE_TIME = auto()
    ARRIVAL_WEIGHTED_ACCURACY = auto()
    DEADLINE_VIOLATIONS = auto()
    ARRIVAL_WEIGHTED_NORMALIZED_DELAY = auto()
    CONSTANT = auto()


@dataclass(frozen=True, slots=True)
class OptimizationPolicy:
    max_components: int
    priority_count: int
    compute_utilization_limit: float
    network_utilization_limit: float
    objective: ObjectiveKind

    def __post_init__(self) -> None:
        for name, count in (
            ("max_components", self.max_components),
            ("priority_count", self.priority_count),
        ):
            if type(count) is not int or count < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name, limit in (
            ("compute_utilization_limit", self.compute_utilization_limit),
            ("network_utilization_limit", self.network_utilization_limit),
        ):
            if not math.isfinite(limit) or not 0 < limit < 1:
                raise ValueError(f"{name} must be strictly between zero and one")
        object.__setattr__(self, "objective", ObjectiveKind(self.objective))


@dataclass(frozen=True, slots=True)
class ModelConfigurationKey:
    model_variant_id: ModelVariantId
    input_shapes: ModelInputConcreteShapes


@dataclass(frozen=True, slots=True)
class VariantData:
    """Graph and costs shared by every input configuration of a variant."""

    accuracies: tuple[PerformanceMetric, ...]
    layers: tuple[LayerKey, ...]
    input_layers: tuple[LayerKey, ...]
    output_layers: tuple[LayerKey, ...]
    edges: tuple[EdgeDescription, ...]
    weight_bytes_by_layer: Mapping[LayerKey, int]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "weight_bytes_by_layer",
            MappingProxyType(dict(self.weight_bytes_by_layer)),
        )


@dataclass(frozen=True, slots=True)
class ConfigurationData:
    batch_size: int
    tensor_bytes: Mapping[TensorKey, int]
    max_cut_size_bytes: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "tensor_bytes", MappingProxyType(dict(self.tensor_bytes))
        )


@dataclass(frozen=True, slots=True)
class WorkerData:
    backend: Backend
    available_inference_memory_bytes: int


@dataclass(frozen=True, slots=True)
class FlowData:
    arrival_rate_per_s: float
    max_response_time_s: float
    performance_requirement: PerformanceMetric
    max_energy_consumption: float
    configurations: tuple[ModelConfigurationKey, ...]
    workers: tuple[WorkerId, ...]
    endpoint_placement: FlowEndpointPlacement
    local_worker_config: LocalWorkerConfig


type ConfigurationWorker = tuple[ModelConfigurationKey, WorkerId]


@dataclass(frozen=True, slots=True)
class CompactOptimizationProblem(OptimizationProblem):
    """Controller projection with exactly one input configuration per declared variant."""

    variants: Mapping[ModelVariantId, VariantData]
    configurations: Mapping[ModelConfigurationKey, ConfigurationData]
    execution_times_s: Mapping[ConfigurationWorker, Mapping[LayerKey, float]]
    flows: Mapping[FlowId, FlowData]
    workers: Mapping[WorkerId, WorkerData]
    connections: tuple[NetworkConnection, ...]
    policy: OptimizationPolicy

    def __post_init__(self) -> None:
        configured_variants = tuple(key.model_variant_id for key in self.configurations)
        if len(configured_variants) != len(set(configured_variants)) or set(
            configured_variants
        ) != set(self.variants):
            raise ValueError(
                "Compact optimization requires exactly one configuration per variant"
            )
        object.__setattr__(self, "variants", MappingProxyType(dict(self.variants)))
        object.__setattr__(
            self, "configurations", MappingProxyType(dict(self.configurations))
        )
        object.__setattr__(
            self,
            "execution_times_s",
            MappingProxyType(
                {
                    key: MappingProxyType(dict(times))
                    for key, times in self.execution_times_s.items()
                }
            ),
        )
        object.__setattr__(self, "flows", MappingProxyType(dict(self.flows)))
        object.__setattr__(self, "workers", MappingProxyType(dict(self.workers)))
