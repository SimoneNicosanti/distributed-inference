import math
from collections.abc import Mapping
from typing import override

from controller.application.optimization.abc.optimization_problem_factory import (
    OptimizationProblemFactory,
)
from controller.domain.optimization.context.optimization_context import (
    OptimizationContext,
)
from controller.domain.optimization.problem.compact_optimization_problem import (
    CompactOptimizationProblem,
    ConfigurationData,
    ConfigurationWorker,
    FlowData,
    ModelConfigurationKey,
    OptimizationPolicy,
    VariantData,
    WorkerData,
)
from controller.domain.optimization.request.flow_endpoint_placement import (
    FixedPlacement,
)
from controller.domain.optimization.request.optimization_request import (
    OptimizationRequest,
    OptimizationRequestsByFlow,
)
from controller.domain.profile.model_variant_static_profile import (
    ModelInputConcreteShapes,
    ModelVariantStaticProfile,
    ShapeProfile,
)
from controller.domain.profile.network_topology import NetworkConnection
from shared.model.keys import LayerKey, TensorKey
from shared.model.model_variant import (
    ModelVariantId,
    PerformanceMetric,
    PerformanceMetricType,
)
from shared.service.service import WorkerId


class CompactOptimizationProblemBuildError(ValueError):
    pass


def _model_key(variant: ModelVariantId) -> tuple[str, str]:
    return variant.model_id.model_name, variant.variant_tag


def _satisfies(metric: PerformanceMetric, requirement: PerformanceMetric) -> bool:
    if metric.type != requirement.type or metric.dataset != requirement.dataset:
        return False
    if not math.isfinite(metric.value) or not math.isfinite(requirement.value):
        return False
    if metric.type in {PerformanceMetricType.MAE, PerformanceMetricType.RMSE}:
        return metric.value <= requirement.value
    return metric.value >= requirement.value


def _eligible(request: OptimizationRequest, profile: ModelVariantStaticProfile) -> bool:
    return profile.info.task == request.task and any(
        _satisfies(metric, request.qos_requirements.min_model_performance)
        for metric in profile.info.accuracies
    )


def _variant_data(profile: ModelVariantStaticProfile) -> VariantData:
    topology = profile.topology
    layers = topology.layers_in_topological_order()
    descriptions = topology.layers()
    inputs = tuple(layer for layer in layers if descriptions[layer].is_input)
    outputs = tuple(layer for layer in layers if descriptions[layer].is_output)
    if not inputs or not outputs:
        raise CompactOptimizationProblemBuildError(
            f"Variant {profile.model_variant_id!r} needs input and output layers"
        )
    weights = {}
    for layer in layers:
        properties = profile.invariant_layer_properties.get(layer)
        weights[layer] = (
            sum(
                math.prod(weight.shape) * weight.data_type_size_in_bytes
                for weight in properties.weight_properties.values()
            )
            if properties is not None
            else 0
        )
    return VariantData(
        accuracies=profile.info.accuracies,
        layers=layers,
        input_layers=inputs,
        output_layers=outputs,
        edges=tuple(edge for _, edge in sorted(topology.edges().items())),
        weight_bytes_by_layer=weights,
    )


def _configuration_data(
    profile: ModelVariantStaticProfile,
    shape: ShapeProfile,
    input_tensors: set[TensorKey],
    tensors: tuple[TensorKey, ...],
) -> ConfigurationData:
    inputs = shape.input_shapes.input_shapes
    if set(inputs) != input_tensors:
        raise CompactOptimizationProblemBuildError(
            f"Input shape keys do not match model inputs for {profile.model_variant_id!r}"
        )
    if not inputs or any(not dimensions for dimensions in inputs.values()):
        raise CompactOptimizationProblemBuildError(
            "Compact requires non-scalar inputs with a shared batch dimension"
        )
    batches = {dimensions[0] for dimensions in inputs.values()}
    if len(batches) != 1:
        raise CompactOptimizationProblemBuildError("Input batch dimensions disagree")
    sizes: dict[TensorKey, int] = {}
    for tensor in tensors:
        invariant = profile.invariant_tensor_properties.get(tensor)
        concrete = shape.tensor_properties.get(tensor)
        if invariant is None or concrete is None:
            raise CompactOptimizationProblemBuildError(
                f"Tensor {tensor!r} has an incomplete static profile"
            )
        if tensor in inputs and concrete.shape != inputs[tensor]:
            raise CompactOptimizationProblemBuildError(
                f"Tensor {tensor!r} shape disagrees with configured input shape"
            )
        sizes[tensor] = math.prod(concrete.shape) * invariant.data_type_size_in_bytes
    maximum = shape.max_cut_size_mib
    if maximum is None or not math.isfinite(maximum) or maximum <= 0:
        raise CompactOptimizationProblemBuildError(
            "Compact requires profiled max_cut_size_mib for every input configuration"
        )
    maximum_bytes = maximum * 1024**2
    if not math.isfinite(maximum_bytes):
        raise CompactOptimizationProblemBuildError("Maximum cut size overflows bytes")
    return ConfigurationData(
        batch_size=next(iter(batches)),
        tensor_bytes=sizes,
        max_cut_size_bytes=math.ceil(maximum_bytes),
    )


def _execution_times(
    context: OptimizationContext,
    variants: Mapping[ModelVariantId, VariantData],
    configurations: Mapping[ModelConfigurationKey, ConfigurationData],
    workers: Mapping[WorkerId, WorkerData],
) -> dict[ConfigurationWorker, Mapping[LayerKey, float]]:
    times: dict[ConfigurationWorker, Mapping[LayerKey, float]] = {
        (key, worker): dict.fromkeys(
            (
                *variants[key.model_variant_id].input_layers,
                *variants[key.model_variant_id].output_layers,
            ),
            0.0,
        )
        for key in configurations
        for worker in workers
    }
    for worker, resource in workers.items():
        worker_profiles = context.execution_profiles_by_worker.get(worker)
        if worker_profiles is None:
            continue
        for variant, profile in worker_profiles.execution_profiles.items():
            if variant not in variants:
                continue
            seen: set[ModelConfigurationKey] = set()
            for shape in profile.shape_execution_profiles:
                key = ModelConfigurationKey(
                    variant, ModelInputConcreteShapes(shape.input_shapes.input_shapes)
                )
                if key not in configurations:
                    raise CompactOptimizationProblemBuildError(
                        "Execution profile references an unknown input configuration"
                    )
                if key in seen:
                    raise CompactOptimizationProblemBuildError(
                        "Duplicate execution profiles for an input configuration"
                    )
                seen.add(key)
                matching = [
                    backend
                    for backend in shape.backend_profiles
                    if backend.backend == resource.backend
                ]
                if len(matching) > 1:
                    raise CompactOptimizationProblemBuildError(
                        f"Duplicate {resource.backend.value} execution profiles"
                    )
                if not matching:
                    continue
                measured = matching[0].layer_profiles
                unknown = set(measured) - set(variants[variant].layers)
                if unknown:
                    raise CompactOptimizationProblemBuildError(
                        f"Execution profile references unknown layers: {sorted(unknown)!r}"
                    )
                values = {
                    layer: float(measurement.execution_time_s)
                    for layer, measurement in measured.items()
                }
                if any(
                    not math.isfinite(value) or value < 0 for value in values.values()
                ):
                    raise CompactOptimizationProblemBuildError(
                        "Execution times must be finite and non-negative"
                    )
                # Boundary layers remain usable without computational profiles.
                times[key, worker] = {**values, **times[key, worker]}
    return times


## TODO: Make this object build more modular
class CompactOptimizationProblemFactory(OptimizationProblemFactory):
    """Project eligible variants, each having exactly one profiled input shape."""

    def __init__(self, policy: OptimizationPolicy) -> None:
        self._policy = policy

    @override
    async def build_optimization_problem(
        self,
        requests_by_flow: OptimizationRequestsByFlow,
        context: OptimizationContext,
    ) -> CompactOptimizationProblem:
        if not requests_by_flow:
            raise CompactOptimizationProblemBuildError(
                "At least one optimization request is required"
            )
        if any(flow != request.flow_id for flow, request in requests_by_flow.items()):
            raise CompactOptimizationProblemBuildError(
                "Optimization request keys do not match their flow IDs"
            )
        worker_ids = tuple(
            sorted(
                context.network_topology.workers()
                & set(context.resource_profiles_by_worker),
                key=lambda worker: str(worker.id),
            )
        )
        if not worker_ids:
            raise CompactOptimizationProblemBuildError(
                "No active worker has a resource profile"
            )
        workers = {
            worker: WorkerData(
                backend=context.resource_profiles_by_worker[worker].execution_backend,
                available_inference_memory_bytes=(
                    context.resource_profiles_by_worker[
                        worker
                    ].available_inference_memory_bytes
                ),
            )
            for worker in worker_ids
        }
        models_by_flow = {}
        for flow, request in sorted(
            requests_by_flow.items(), key=lambda item: str(item[0])
        ):
            placement = request.flow_endpoint_placement
            for endpoint in (placement.ingress_placement, placement.sink_placement):
                if (
                    isinstance(endpoint, FixedPlacement)
                    and endpoint.worker_id not in workers
                ):
                    raise CompactOptimizationProblemBuildError(
                        "Fixed endpoint worker is inactive or lacks a resource profile"
                    )
            eligible = tuple(
                sorted(
                    (
                        variant
                        for variant, profile in context.static_profiles_by_variant.items()
                        if _eligible(request, profile)
                    ),
                    key=_model_key,
                )
            )
            if not eligible:
                raise CompactOptimizationProblemBuildError(
                    f"Flow {flow!r} has no eligible model variant"
                )
            models_by_flow[flow] = eligible

        variants: dict[ModelVariantId, VariantData] = {}
        configurations: dict[ModelConfigurationKey, ConfigurationData] = {}
        selected = {variant for models in models_by_flow.values() for variant in models}
        for variant in sorted(selected, key=_model_key):
            profile = context.static_profiles_by_variant[variant]
            if len(profile.shape_profiles) != 1:
                raise CompactOptimizationProblemBuildError(
                    f"Variant {variant!r} has {len(profile.shape_profiles)} profiled "
                    "input configurations; compact optimization requires exactly "
                    "one configuration per variant"
                )
            data = _variant_data(profile)
            variants[variant] = data
            input_tensors = {
                tensor
                for edge in data.edges
                if edge.source in data.input_layers
                for tensor in edge.tensors
            }
            tensors = tuple(sorted(profile.topology.tensor_routes()))
            shape = profile.shape_profiles[0]
            key = ModelConfigurationKey(
                variant, ModelInputConcreteShapes(shape.input_shapes.input_shapes)
            )
            configurations[key] = _configuration_data(
                profile, shape, input_tensors, tensors
            )

        flows = {
            flow: FlowData(
                arrival_rate_per_s=float(
                    requests_by_flow[flow].stream_info.arrival_rate
                ),
                max_response_time_s=float(
                    requests_by_flow[flow].qos_requirements.max_mean_response_time
                ),
                performance_requirement=requests_by_flow[
                    flow
                ].qos_requirements.min_model_performance,
                max_energy_consumption=requests_by_flow[
                    flow
                ].qos_requirements.max_energy_consumption,
                configurations=tuple(
                    key for key in configurations if key.model_variant_id in models
                ),
                workers=worker_ids,
                endpoint_placement=requests_by_flow[flow].flow_endpoint_placement,
                local_worker_config=requests_by_flow[flow].local_worker_config,
            )
            for flow, models in models_by_flow.items()
        }
        connections = tuple(
            NetworkConnection(
                connection.source,
                connection.target,
                connection.throughput_bytes_per_sec,
                connection.round_trip_time_s,
            )
            for _, connection in sorted(
                context.network_topology.connections().items(),
                key=lambda item: (str(item[0][0].id), str(item[0][1].id)),
            )
            if connection.source in workers and connection.target in workers
        )
        return CompactOptimizationProblem(
            variants=variants,
            configurations=configurations,
            execution_times_s=_execution_times(
                context, variants, configurations, workers
            ),
            flows=flows,
            workers=workers,
            connections=connections,
            policy=self._policy,
        )
