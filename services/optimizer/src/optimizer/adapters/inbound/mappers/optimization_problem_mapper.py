from integration.dto.optimization.optimization_dto import (
    AnyPlacementDto,
    FixedPlacementDto,
    LocalWorkerConfigDto,
    NoLocalWorkerConfigDto,
    PlacementDto,
    ValidLocalWorkerConfigDto,
)
from integration.dto.optimizer.common import PerformanceMetricDto
from integration.dto.optimizer.optimization_problem_dto import (
    ModelConfigurationKeyDto,
    OptimizationProblemDto,
)
from integration.shared.id_json import service_id_from_json, service_id_to_json
from optimizer.domain.problem.other.backend import Backend
from optimizer.domain.problem.compact.compact_optimization_problem import (
    CompactOptimizationProblem,
    ConfigurationData,
    FlowData,
    ModelConfigurationKey,
    ObjectiveKind,
    OptimizationPolicy,
    VariantData,
    WorkerData,
)
from optimizer.domain.problem.other.edge_description import EdgeDescription
from optimizer.domain.problem.other.model_input_concrete_shapes import (
    ModelInputConcreteShapes,
)
from optimizer.domain.problem.other.network_connection import NetworkConnection
from optimizer.domain.problem.other.flow_endpoint_placement import (
    AnyPlacement,
    FixedPlacement,
    FlowEndpointPlacement,
    Placement,
)
from optimizer.domain.problem.other.local_worker_config import (
    LocalWorkerConfig,
    NoLocalWorkerConfig,
    ValidLocalWorkerConfig,
)
from shared.model.model_variant import PerformanceMetric


def _metric(metric: PerformanceMetricDto) -> PerformanceMetric:
    return PerformanceMetric(metric.type, metric.dataset, metric.value)


def _key(key: ModelConfigurationKeyDto) -> ModelConfigurationKey:
    return ModelConfigurationKey(
        key.model_variant_id, ModelInputConcreteShapes(key.input_shapes.input_shapes)
    )


def _placement(placement: PlacementDto) -> Placement:
    match placement:
        case FixedPlacementDto():
            return FixedPlacement(
                service_id_from_json(service_id_to_json(placement.worker_id))
            )
        case AnyPlacementDto():
            return AnyPlacement()
    raise TypeError(f"Unsupported placement: {type(placement).__name__}")


def _local(local: LocalWorkerConfigDto) -> LocalWorkerConfig:
    match local:
        case ValidLocalWorkerConfigDto():
            return ValidLocalWorkerConfig(
                service_id_from_json(service_id_to_json(local.worker_id)),
                local.is_inference_runner,
            )
        case NoLocalWorkerConfigDto():
            return NoLocalWorkerConfig()
    raise TypeError(f"Unsupported local worker configuration: {type(local).__name__}")


class OptimizationProblemMapper:
    @staticmethod
    def from_dto(dto: OptimizationProblemDto) -> CompactOptimizationProblem:
        return CompactOptimizationProblem(
            variants={
                data.model_variant_id: VariantData(
                    accuracies=tuple(_metric(metric) for metric in data.accuracies),
                    layers=data.layers,
                    input_layers=data.input_layers,
                    output_layers=data.output_layers,
                    edges=tuple(
                        EdgeDescription(
                            edge.source, edge.target, frozenset(edge.tensors)
                        )
                        for edge in data.edges
                    ),
                    weight_bytes_by_layer=data.weight_bytes_by_layer,
                )
                for data in dto.variants
            },
            configurations={
                _key(data.key): ConfigurationData(
                    data.batch_size, data.tensor_bytes, data.max_cut_size_bytes
                )
                for data in dto.configurations
            },
            execution_times_s={
                (_key(cost.configuration), cost.worker_id): cost.layer_times_s
                for cost in dto.execution_times_s
            },
            flows={
                flow.flow_id: FlowData(
                    arrival_rate_per_s=flow.arrival_rate_per_s,
                    max_response_time_s=flow.max_response_time_s,
                    performance_requirement=_metric(flow.performance_requirement),
                    max_energy_consumption=flow.max_energy_consumption,
                    configurations=tuple(_key(key) for key in flow.configurations),
                    workers=flow.workers,
                    endpoint_placement=FlowEndpointPlacement(
                        _placement(flow.endpoint_placement.ingress_placement),
                        _placement(flow.endpoint_placement.sink_placement),
                    ),
                    local_worker_config=_local(flow.local_worker_config),
                )
                for flow in dto.flows
            },
            workers={
                data.worker_id: WorkerData(
                    Backend(data.backend.value), data.available_inference_memory_bytes
                )
                for data in dto.workers
            },
            connections=tuple(
                NetworkConnection(
                    link.source,
                    link.target,
                    link.throughput_bytes_per_sec,
                    link.round_trip_time_s,
                )
                for link in dto.connections
            ),
            policy=OptimizationPolicy(
                max_components=dto.policy.max_components,
                priority_count=dto.policy.priority_count,
                compute_utilization_limit=dto.policy.compute_utilization_limit,
                network_utilization_limit=dto.policy.network_utilization_limit,
                objective=ObjectiveKind(dto.policy.objective.value),
            ),
        )
