from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from optimizer.domain.formulation.formulation import Formulation
from optimizer.domain.problem.compact.compact_optimization_problem import (
    CompactOptimizationProblem,
    ModelConfigurationKey,
)
from optimizer.domain.problem.other.network_connection import NetworkConnection
from shared.flow.flow import FlowId
from shared.model.keys import LayerKey, TensorKey
from shared.model.model_variant import ModelVariantId
from shared.service.service import WorkerId

type FlowConfiguration = tuple[FlowId, ModelConfigurationKey]
type VariantLayer = tuple[ModelVariantId, LayerKey]
type VariantTensor = tuple[ModelVariantId, TensorKey]


@dataclass(frozen=True, slots=True)
class TensorRoute:
    source: LayerKey
    destinations: tuple[LayerKey, ...]
    maximal_destinations: tuple[LayerKey, ...]


@dataclass(frozen=True, slots=True)
class MisocpCompactFormulation(Formulation):
    """Prepared MISOCP coefficients and indices, independent of solver APIs."""

    problem: CompactOptimizationProblem
    flows: tuple[FlowId, ...]
    workers: tuple[WorkerId, ...]
    configurations_by_flow: Mapping[FlowId, tuple[ModelConfigurationKey, ...]]
    service_arrival_rate_per_s: Mapping[FlowConfiguration, float]
    batch_fill_time_s: Mapping[FlowConfiguration, float]
    compute_bin_limits: tuple[float, ...]
    network_bin_limits: tuple[float, ...]
    max_queue_wait_s_by_worker: Mapping[WorkerId, float]
    max_queue_wait_s_by_flow: Mapping[FlowId, float]
    flow_index: Mapping[FlowId, int]
    worker_index: Mapping[WorkerId, int]
    configuration_index: Mapping[FlowConfiguration, int]
    layer_index: Mapping[VariantLayer, int]
    tensor_index: Mapping[VariantTensor, int]
    tensor_routes_by_variant: Mapping[ModelVariantId, Mapping[TensorKey, TensorRoute]]
    connections_by_pair: Mapping[tuple[WorkerId, WorkerId], NetworkConnection]
    successors_by_worker: Mapping[WorkerId, tuple[WorkerId, ...]]
    predecessors_by_worker: Mapping[WorkerId, tuple[WorkerId, ...]]

    def __post_init__(self) -> None:
        for name in (
            "configurations_by_flow",
            "service_arrival_rate_per_s",
            "batch_fill_time_s",
            "max_queue_wait_s_by_worker",
            "max_queue_wait_s_by_flow",
            "flow_index",
            "worker_index",
            "configuration_index",
            "layer_index",
            "tensor_index",
            "connections_by_pair",
            "successors_by_worker",
            "predecessors_by_worker",
        ):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))
        object.__setattr__(
            self,
            "tensor_routes_by_variant",
            MappingProxyType(
                {
                    variant: MappingProxyType(dict(routes))
                    for variant, routes in self.tensor_routes_by_variant.items()
                }
            ),
        )
