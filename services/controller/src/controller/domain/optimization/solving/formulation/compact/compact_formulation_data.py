from collections.abc import Mapping
from dataclasses import dataclass

from controller.domain.optimization.request.flow_endpoint_placement import (
    FlowEndpointPlacement,
)
from controller.domain.optimization.request.local_worker_config import LocalWorkerConfig
from controller.domain.optimization.solving.formulation.compact.compact_objective_kind import (
    CompactObjectiveKind,
)
from controller.domain.profile.model_variant_execution_profile import (
    LayerExecutionProfile,
)
from controller.domain.profile.model_variant_static_profile import (
    ModelVariantStaticProfile,
)
from controller.domain.profile.network_topology import NetworkTopology
from controller.domain.profile.resource_profile import WorkerResourceProfile
from shared.flow.flow import FlowId
from shared.model.keys import LayerKey, TensorKey
from shared.model.model_variant import ModelVariantId, PerformanceMetric
from shared.service.service import WorkerId

type FlowModel = tuple[FlowId, ModelVariantId]
type ModelLayer = tuple[ModelVariantId, LayerKey]
type ModelTensor = tuple[ModelVariantId, TensorKey]


@dataclass(frozen=True, slots=True)
class CompactFormulationData:
    """Complete inputs for the compact formulation, independent of solver APIs."""

    max_components: int
    priority_count: int
    compute_utilization_limit: float
    network_utilization_limit: float
    objective_kind: CompactObjectiveKind
    static_profiles_by_variant: Mapping[ModelVariantId, ModelVariantStaticProfile]
    resource_profiles_by_worker: Mapping[WorkerId, WorkerResourceProfile]
    network_topology: NetworkTopology
    layer_execution_profiles_by_model_worker: Mapping[
        tuple[ModelVariantId, WorkerId], Mapping[LayerKey, LayerExecutionProfile] | None
    ]
    arrival_rate_per_s: dict[FlowId, float]
    max_response_time_s_by_flow: dict[FlowId, float]
    performance_requirement_by_flow: dict[FlowId, PerformanceMetric]
    endpoint_placement_by_flow: dict[FlowId, FlowEndpointPlacement]
    local_worker_config_by_flow: dict[FlowId, LocalWorkerConfig]
    # Ordered and filtered index sets used by the formulation.
    flows: tuple[FlowId, ...]
    workers: tuple[WorkerId, ...]
    workers_by_flow: dict[FlowId, tuple[WorkerId, ...]]
    models_by_flow: dict[FlowId, tuple[ModelVariantId, ...]]

    # Coefficients derived from multiple domain inputs.
    service_arrival_rate_per_s: dict[FlowModel, float]
    batch_fill_time_s: dict[FlowModel, float]
    compute_bin_limits: tuple[float, ...]
    network_bin_limits: tuple[float, ...]
    max_queue_wait_s_by_worker: dict[WorkerId, float]
    max_queue_wait_s_by_flow: dict[FlowId, float]

    # Stable numeric indices for variables and constraints.
    flow_index: dict[FlowId, int]
    worker_index: dict[WorkerId, int]
    model_index: dict[FlowModel, int]
    layer_index: dict[ModelLayer, int]
    tensor_index: dict[ModelTensor, int]
