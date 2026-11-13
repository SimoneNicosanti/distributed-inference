from controller.domain.optimization.problem.compact_optimization_problem import (
    CompactOptimizationProblem,
    ModelConfigurationKey,
)
from controller.domain.optimization.problem.optimization_problem import (
    OptimizationProblem,
)
from controller.domain.optimization.request.flow_endpoint_placement import (
    AnyPlacement,
    FixedPlacement,
    Placement,
)
from controller.domain.optimization.request.local_worker_config import (
    LocalWorkerConfig,
    NoLocalWorkerConfig,
    ValidLocalWorkerConfig,
)
from integration.dto.model.profiled_model_variant_dto import ModelInputConcreteShapesDto
from integration.dto.optimization.optimization_dto import (
    AnyPlacementDto,
    FixedPlacementDto,
    FlowEndpointPlacementDto,
    LocalWorkerConfigDto,
    NoLocalWorkerConfigDto,
    PlacementDto,
    ValidLocalWorkerConfigDto,
)
from integration.dto.optimizer.common import PerformanceMetricDto
from integration.dto.optimizer.optimization_problem_dto import (
    CompactOptimizationProblemDto,
    ConfigurationDataDto,
    ConnectionDataDto,
    EdgeDataDto,
    ExecutionTimesDto,
    FlowDataDto,
    ModelConfigurationKeyDto,
    ObjectiveKindDto,
    OptimizationPolicyDto,
    OptimizationProblemDto,
    VariantDataDto,
    WorkerDataDto,
)
from integration.dto.worker_profiles.execution_profile_dto import BackendDto
from shared.model.model_variant import PerformanceMetric


def _metric(metric: PerformanceMetric) -> PerformanceMetricDto:
    return PerformanceMetricDto(
        type=metric.type, dataset=metric.dataset, value=metric.value
    )


def _key(key: ModelConfigurationKey) -> ModelConfigurationKeyDto:
    return ModelConfigurationKeyDto(
        model_variant_id=key.model_variant_id,
        input_shapes=ModelInputConcreteShapesDto(
            input_shapes=dict(key.input_shapes.input_shapes)
        ),
    )


def _placement(placement: Placement) -> PlacementDto:
    match placement:
        case FixedPlacement():
            return FixedPlacementDto(worker_id=placement.worker_id)
        case AnyPlacement():
            return AnyPlacementDto()
    raise TypeError(f"Unsupported placement: {type(placement).__name__}")


def _local(local: LocalWorkerConfig) -> LocalWorkerConfigDto:
    match local:
        case ValidLocalWorkerConfig():
            return ValidLocalWorkerConfigDto(
                worker_id=local.worker_id, is_inference_runner=local.is_inference_runner
            )
        case NoLocalWorkerConfig():
            return NoLocalWorkerConfigDto()
    raise TypeError(f"Unsupported local worker configuration: {type(local).__name__}")


class OptimizationProblemMapper:
    @staticmethod
    def to_dto(problem: OptimizationProblem) -> OptimizationProblemDto:
        if not isinstance(problem, CompactOptimizationProblem):
            raise TypeError(
                f"Unsupported optimization problem: {type(problem).__name__}"
            )
        return CompactOptimizationProblemDto(
            variants=tuple(
                VariantDataDto(
                    model_variant_id=variant_id,
                    accuracies=tuple(_metric(metric) for metric in data.accuracies),
                    layers=data.layers,
                    input_layers=data.input_layers,
                    output_layers=data.output_layers,
                    edges=tuple(
                        EdgeDataDto(
                            source=edge.source,
                            target=edge.target,
                            tensors=tuple(sorted(edge.tensors)),
                        )
                        for edge in data.edges
                    ),
                    weight_bytes_by_layer=dict(data.weight_bytes_by_layer),
                )
                for variant_id, data in problem.variants.items()
            ),
            configurations=tuple(
                ConfigurationDataDto(
                    key=_key(key),
                    batch_size=data.batch_size,
                    tensor_bytes=dict(data.tensor_bytes),
                    max_cut_size_bytes=data.max_cut_size_bytes,
                )
                for key, data in problem.configurations.items()
            ),
            execution_times_s=tuple(
                ExecutionTimesDto(
                    configuration=_key(key), worker_id=worker, layer_times_s=dict(times)
                )
                for (key, worker), times in problem.execution_times_s.items()
            ),
            flows=tuple(
                FlowDataDto(
                    flow_id=flow_id,
                    arrival_rate_per_s=data.arrival_rate_per_s,
                    max_response_time_s=data.max_response_time_s,
                    performance_requirement=_metric(data.performance_requirement),
                    max_energy_consumption=data.max_energy_consumption,
                    configurations=tuple(_key(key) for key in data.configurations),
                    workers=data.workers,
                    endpoint_placement=FlowEndpointPlacementDto(
                        ingress_placement=_placement(
                            data.endpoint_placement.ingress_placement
                        ),
                        sink_placement=_placement(
                            data.endpoint_placement.sink_placement
                        ),
                    ),
                    local_worker_config=_local(data.local_worker_config),
                )
                for flow_id, data in problem.flows.items()
            ),
            workers=tuple(
                WorkerDataDto(
                    worker_id=worker,
                    backend=BackendDto(data.backend.value),
                    available_inference_memory_bytes=data.available_inference_memory_bytes,
                )
                for worker, data in problem.workers.items()
            ),
            connections=tuple(
                ConnectionDataDto(
                    source=link.source,
                    target=link.target,
                    throughput_bytes_per_sec=link.throughput_bytes_per_sec,
                    round_trip_time_s=link.round_trip_time_s,
                )
                for link in problem.connections
            ),
            policy=OptimizationPolicyDto(
                max_components=problem.policy.max_components,
                priority_count=problem.policy.priority_count,
                compute_utilization_limit=problem.policy.compute_utilization_limit,
                network_utilization_limit=problem.policy.network_utilization_limit,
                objective=ObjectiveKindDto(problem.policy.objective.value),
            ),
        )
