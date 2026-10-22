import math
from abc import ABC, abstractmethod
from typing import override

from controller.adapters.outbound.optimization.gurobi_problem import (
    GurobiProblem,
)
from controller.adapters.outbound.optimization.gurobi_profile_data import (
    GurobiProfileDataError,
    batch_size,
    fixed_shape_profile,
    layer_execution_profiles,
    layer_weight_size_bytes,
    tensor_size_bytes,
)
from controller.domain.optimization.optimization_context import OptimizationContext
from controller.domain.optimization.optimization_request import (
    OptimizationRequest,
    OptimizationRequestsByFlow,
)
from controller.domain.optimization.solver_config import ProblemConfig
from controller.domain.profile.model_variant_static_profile import (
    ModelVariantStaticProfile,
)
from integration.shared.id_json import flow_id_to_json, model_variant_id_to_json
from shared.flow.flow import FlowId
from shared.model.keys import LayerKey, TensorKey
from shared.model.model import ModelVisibility
from shared.model.model_variant import (
    ModelVariantId,
    PerformanceMetric,
    PerformanceMetricType,
)
from shared.service.service import WorkerId


class GurobiProblemBuildError(ValueError):
    pass


def model_key(model_variant_id: ModelVariantId) -> str:
    return model_variant_id_to_json(model_variant_id)


def flow_key(flow_id: FlowId) -> str:
    return flow_id_to_json(flow_id)


def worker_key(worker_id: WorkerId) -> str:
    return str(worker_id.service_id)


def _is_metric_satisfied(
    metric: PerformanceMetric,
    requirement: PerformanceMetric,
) -> bool:
    if metric.type != requirement.type or metric.dataset != requirement.dataset:
        return False
    if not math.isfinite(metric.value) or not math.isfinite(requirement.value):
        return False
    if metric.type in {PerformanceMetricType.MAE, PerformanceMetricType.RMSE}:
        return metric.value <= requirement.value
    if metric.type in {
        PerformanceMetricType.MAP_AT_50,
        PerformanceMetricType.MAP_AT_75,
    }:
        return metric.value >= requirement.value
    raise GurobiProblemBuildError(
        f"Unsupported performance metric type: {metric.type!r}"
    )


def _is_model_eligible(
    request: OptimizationRequest,
    profile: ModelVariantStaticProfile,
) -> bool:
    if profile.info.task != request.task:
        return False
    if (
        profile.info.visibility == ModelVisibility.PRIVATE
        and profile.model_variant_id.model_id.owner_id != request.flow_id.user_id
    ):
        return False
    return any(
        _is_metric_satisfied(metric, request.qos_requirements.min_model_performance)
        for metric in profile.info.accuracies
    )


def _bin_limits(utilization_limit: float, relative_error: float) -> tuple[float, ...]:
    interval_count = math.ceil(
        -math.log(1 - utilization_limit) / math.log(1 + relative_error)
    )
    limits = [0.0]
    limits.extend(
        1 - 1 / (1 + relative_error) ** index for index in range(1, interval_count)
    )
    limits.append(utilization_limit)
    return tuple(limits)


def _eligible_models_by_flow(
    flows: tuple[FlowId, ...],
    requests_by_flow: OptimizationRequestsByFlow,
    context: OptimizationContext,
) -> dict[FlowId, tuple[ModelVariantId, ...]]:
    result: dict[FlowId, tuple[ModelVariantId, ...]] = {}
    for flow in flows:
        request = requests_by_flow[flow]
        eligible = tuple(
            sorted(
                (
                    model_variant_id
                    for model_variant_id, profile in (
                        context.static_profiles_by_variant.items()
                    )
                    if _is_model_eligible(request, profile)
                ),
                key=model_key,
            )
        )
        if not eligible:
            raise GurobiProblemBuildError(
                f"Flow {flow_key(flow)} has no eligible model variant"
            )
        result[flow] = eligible
    return result


def _workers_by_flow(
    flows: tuple[FlowId, ...],
    requests_by_flow: OptimizationRequestsByFlow,
    context: OptimizationContext,
) -> tuple[tuple[WorkerId, ...], dict[FlowId, tuple[WorkerId, ...]]]:
    workers = tuple(
        sorted(
            context.network_topology.workers()
            & set(context.resource_profiles_by_worker),
            key=worker_key,
        )
    )
    if not workers:
        raise GurobiProblemBuildError("No active worker has a resource profile")

    private_devices = {request.ingress_id for request in requests_by_flow.values()}
    result: dict[FlowId, tuple[WorkerId, ...]] = {}
    for flow in flows:
        ingress = requests_by_flow[flow].ingress_id
        if ingress not in workers:
            raise GurobiProblemBuildError(
                f"Ingress worker {worker_key(ingress)} is inactive or lacks a "
                "resource profile"
            )
        result[flow] = tuple(
            worker
            for worker in workers
            if worker == ingress or worker not in private_devices
        )
    return workers, result


def _build_gurobi_problem(
    requests_by_flow: OptimizationRequestsByFlow,
    context: OptimizationContext,
    *,
    compute_utilization_limit: float,
    network_utilization_limit: float,
    relative_error: float,
) -> GurobiProblem:
    if not requests_by_flow:
        raise GurobiProblemBuildError("At least one optimization request is required")

    mismatched_flows = tuple(
        flow for flow, request in requests_by_flow.items() if request.flow_id != flow
    )
    if mismatched_flows:
        raise GurobiProblemBuildError(
            "Optimization request keys do not match their flow IDs: "
            f"{[flow_key(flow) for flow in mismatched_flows]!r}"
        )
    flows = tuple(sorted(requests_by_flow, key=flow_key))
    workers, workers_by_flow = _workers_by_flow(flows, requests_by_flow, context)
    models_by_flow = _eligible_models_by_flow(flows, requests_by_flow, context)
    selected_models = {model for models in models_by_flow.values() for model in models}

    layers_by_model: dict[ModelVariantId, tuple[LayerKey, ...]] = {}
    tensors_by_model: dict[ModelVariantId, tuple[TensorKey, ...]] = {}
    batch_sizes: dict[ModelVariantId, int] = {}

    for model in sorted(selected_models, key=model_key):
        profile = context.static_profiles_by_variant[model]
        fixed_shape_profile(profile)
        batch_sizes[model] = batch_size(profile)

        layers = profile.topology.layers_in_topological_order()
        descriptions = profile.topology.layers()
        inputs = tuple(layer for layer in layers if descriptions[layer].is_input)
        outputs = tuple(layer for layer in layers if descriptions[layer].is_output)
        if not inputs or not outputs:
            raise GurobiProblemBuildError(
                f"Model variant {model_key(model)} needs input and output layers"
            )
        layers_by_model[model] = layers
        routes = profile.topology.tensor_routes()
        tensors = tuple(routes)
        tensors_by_model[model] = tensors
        for tensor in tensors:
            tensor_size_bytes(profile, tensor)
        for layer in layers:
            layer_weight_size_bytes(profile, layer)

    arrival_rate = {
        flow: float(requests_by_flow[flow].stream_info.arrival_rate) for flow in flows
    }
    service_arrival_rate = {
        (flow, model): arrival_rate[flow] / batch_sizes[model]
        for flow in flows
        for model in models_by_flow[flow]
    }
    batch_fill_time = {
        (flow, model): (batch_sizes[model] - 1) / (2 * arrival_rate[flow])
        for flow in flows
        for model in models_by_flow[flow]
    }

    for worker in workers:
        for model in selected_models:
            layer_profiles = layer_execution_profiles(context, model, worker)
            if layer_profiles is None:
                continue
            unknown_layers = set(layer_profiles) - set(layers_by_model[model])
            if unknown_layers:
                raise GurobiProblemBuildError(
                    f"Execution profile contains unknown layers: "
                    f"{sorted(unknown_layers)!r}"
                )
            for layer_profile in layer_profiles.values():
                value = float(layer_profile.execution_time_s)
                if not math.isfinite(value):
                    raise GurobiProblemBuildError("Execution times must be finite")

    max_queue_wait_by_worker: dict[WorkerId, float] = {}
    for worker in workers:
        numerator = 0.0
        for flow in flows:
            if worker not in workers_by_flow[flow]:
                continue
            model_second_moments: list[float] = []
            for model in models_by_flow[flow]:
                profiles = layer_execution_profiles(context, model, worker) or {}
                total_execution_time = sum(
                    float(profiles[layer].execution_time_s)
                    for layer in layers_by_model[model]
                    if layer in profiles
                )
                model_second_moments.append(
                    service_arrival_rate[flow, model] * total_execution_time**2
                )
            numerator += max(model_second_moments)
        max_queue_wait_by_worker[worker] = numerator / (
            2 * (1 - compute_utilization_limit) ** 2
        )
    max_queue_wait_by_flow = {
        flow: max(max_queue_wait_by_worker[worker] for worker in workers_by_flow[flow])
        for flow in flows
    }

    flow_index = {flow: index for index, flow in enumerate(flows)}
    worker_index = {worker: index for index, worker in enumerate(workers)}
    model_index = {
        (flow, model): index
        for flow in flows
        for index, model in enumerate(models_by_flow[flow])
    }
    layer_index = {
        (model, layer): index
        for model in selected_models
        for index, layer in enumerate(layers_by_model[model])
    }
    tensor_index = {
        (model, tensor): index
        for model in selected_models
        for index, tensor in enumerate(tensors_by_model[model])
    }

    return GurobiProblem(
        flows=flows,
        workers=workers,
        workers_by_flow=workers_by_flow,
        models_by_flow=models_by_flow,
        service_arrival_rate=service_arrival_rate,
        batch_fill_time=batch_fill_time,
        compute_bin_limits=_bin_limits(compute_utilization_limit, relative_error),
        network_bin_limits=_bin_limits(network_utilization_limit, relative_error),
        max_queue_wait_by_worker=max_queue_wait_by_worker,
        max_queue_wait_by_flow=max_queue_wait_by_flow,
        flow_index=flow_index,
        worker_index=worker_index,
        model_index=model_index,
        layer_index=layer_index,
        tensor_index=tensor_index,
    )


class GurobiProblemBuilder(ABC):
    @abstractmethod
    def build(
        self,
        requests_by_flow: OptimizationRequestsByFlow,
        context: OptimizationContext,
        problem_config: ProblemConfig,
    ) -> GurobiProblem: ...


class DefaultGurobiProblemBuilder(GurobiProblemBuilder):
    @override
    def build(
        self,
        requests_by_flow: OptimizationRequestsByFlow,
        context: OptimizationContext,
        problem_config: ProblemConfig,
    ) -> GurobiProblem:
        try:
            return _build_gurobi_problem(
                requests_by_flow,
                context,
                compute_utilization_limit=problem_config.compute_utilization_limit,
                network_utilization_limit=problem_config.network_utilization_limit,
                relative_error=problem_config.relative_error,
            )
        except GurobiProfileDataError as error:
            raise GurobiProblemBuildError(str(error)) from error
