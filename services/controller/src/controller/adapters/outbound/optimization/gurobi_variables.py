from dataclasses import dataclass

import gurobipy as gp
from gurobipy import GRB

from controller.adapters.outbound.optimization.gurobi_problem import (
    FlowModel,
    GurobiProblem,
)
from controller.adapters.outbound.optimization.gurobi_profile_data import tensor_routes
from controller.domain.optimization.optimization_context import OptimizationContext
from controller.domain.optimization.optimization_request import (
    OptimizationRequestsByFlow,
)
from shared.flow.flow import FlowId
from shared.model.keys import LayerKey, TensorKey
from shared.model.model_variant import ModelVariantId
from shared.service.service import WorkerId


@dataclass(slots=True)
class AssignmentVariables:
    model_selected: dict[FlowModel, gp.Var]
    layer_component: dict[tuple[FlowId, ModelVariantId, LayerKey, int], gp.Var]
    component_worker: dict[tuple[FlowId, int, WorkerId], gp.Var]
    component_active: dict[tuple[FlowId, int], gp.Var]
    component_last: dict[tuple[FlowId, int], gp.Var]
    layer_component_worker: dict[
        tuple[FlowId, ModelVariantId, LayerKey, int, WorkerId], gp.Var
    ]


@dataclass(slots=True)
class ExecutionVariables:
    # Its square enters the compute-queue SOC, so it remains explicit.
    execution_time: dict[tuple[FlowId, ModelVariantId, int, WorkerId], gp.Var]


@dataclass(slots=True)
class ComputeQueueVariables:
    component_priority: dict[tuple[FlowId, int, WorkerId, int], gp.Var]
    priority_used: dict[tuple[WorkerId, int], gp.Var]
    priority_class_utilization: dict[tuple[FlowId, int, WorkerId, int], gp.Var]
    lower_priority_utilization: dict[tuple[WorkerId, int], gp.Var]
    compute_bin_selected: dict[tuple[WorkerId, int, int], gp.Var]
    queue_wait_bin: dict[tuple[WorkerId, int, int], gp.Var]
    queue_wait: dict[tuple[WorkerId, int], gp.Var]
    weighted_queue_wait: dict[tuple[WorkerId, int], gp.Var]
    remaining_compute_capacity: dict[tuple[WorkerId, int], gp.Var]
    component_queue_wait: dict[tuple[FlowId, int], gp.Var]


@dataclass(slots=True)
class NetworkVariables:
    tensor_crosses_cut: dict[tuple[FlowId, ModelVariantId, TensorKey, int], gp.Var]
    adjacent_component_workers: dict[tuple[FlowId, int, WorkerId, WorkerId], gp.Var]
    transmission_time: dict[tuple[FlowId, ModelVariantId, int, WorkerId], gp.Var]
    network_utilization: dict[WorkerId, gp.Var]
    network_bin_selected: dict[tuple[WorkerId, int], gp.Var]
    component_transmission_response: dict[tuple[FlowId, int], gp.Var]


@dataclass(slots=True)
class MemoryVariables:
    max_activation_bytes: dict[WorkerId, gp.Var]


@dataclass(slots=True)
class QoSVariables:
    response_time: dict[FlowId, gp.Var]


@dataclass(slots=True)
class GurobiVariables:
    assignment: AssignmentVariables
    execution: ExecutionVariables
    compute_queue: ComputeQueueVariables
    network: NetworkVariables
    memory: MemoryVariables
    qos: QoSVariables


def _add_assignment_variables(
    model: gp.Model,
    problem: GurobiProblem,
    context: OptimizationContext,
    *,
    max_components: int,
) -> AssignmentVariables:
    model_selected: dict[FlowModel, gp.Var] = {}
    layer_component: dict[tuple[FlowId, ModelVariantId, LayerKey, int], gp.Var] = {}
    component_worker: dict[tuple[FlowId, int, WorkerId], gp.Var] = {}
    component_active: dict[tuple[FlowId, int], gp.Var] = {}
    component_last: dict[tuple[FlowId, int], gp.Var] = {}
    layer_component_worker: dict[
        tuple[FlowId, ModelVariantId, LayerKey, int, WorkerId], gp.Var
    ] = {}

    components = range(1, max_components + 1)
    for flow in problem.flows:
        flow_index = problem.flow_index[flow]
        for model_variant in problem.models_by_flow[flow]:
            model_index = problem.model_index[flow, model_variant]
            static_profile = context.static_profiles_by_variant[model_variant]
            model_selected[flow, model_variant] = model.addVar(
                lb=0.0,
                ub=1.0,
                name=f"u[{flow_index},{model_index}]",
            )
            for layer in static_profile.topology.layers_in_topological_order():
                layer_index = problem.layer_index[model_variant, layer]
                for component in components:
                    layer_component[flow, model_variant, layer, component] = (
                        model.addVar(
                            vtype=GRB.BINARY,
                            name=(
                                f"x[{flow_index},{model_index},{layer_index},"
                                f"{component}]"
                            ),
                        )
                    )
                    for worker in problem.workers_by_flow[flow]:
                        worker_index = problem.worker_index[worker]
                        layer_component_worker[
                            flow, model_variant, layer, component, worker
                        ] = model.addVar(
                            lb=0.0,
                            ub=1.0,
                            name=(
                                f"z[{flow_index},{model_index},{layer_index},"
                                f"{component},{worker_index}]"
                            ),
                        )

        for component in components:
            component_active[flow, component] = model.addVar(
                lb=0.0,
                ub=1.0,
                name=f"active[{flow_index},{component}]",
            )
            component_last[flow, component] = model.addVar(
                lb=0.0,
                ub=1.0,
                name=f"last[{flow_index},{component}]",
            )
            for worker in problem.workers_by_flow[flow]:
                component_worker[flow, component, worker] = model.addVar(
                    lb=0.0,
                    ub=1.0,
                    name=(
                        f"e[{flow_index},{component},{problem.worker_index[worker]}]"
                    ),
                )

    return AssignmentVariables(
        model_selected=model_selected,
        layer_component=layer_component,
        component_worker=component_worker,
        component_active=component_active,
        component_last=component_last,
        layer_component_worker=layer_component_worker,
    )


def _add_execution_variables(
    model: gp.Model,
    problem: GurobiProblem,
    *,
    max_components: int,
) -> ExecutionVariables:
    execution_time: dict[tuple[FlowId, ModelVariantId, int, WorkerId], gp.Var] = {}
    for flow in problem.flows:
        flow_index = problem.flow_index[flow]
        for model_variant in problem.models_by_flow[flow]:
            model_index = problem.model_index[flow, model_variant]
            for component in range(1, max_components + 1):
                for worker in problem.workers_by_flow[flow]:
                    worker_index = problem.worker_index[worker]
                    execution_time[flow, model_variant, component, worker] = (
                        model.addVar(
                            lb=0.0,
                            name=(
                                f"tex[{flow_index},{model_index},{component},"
                                f"{worker_index}]"
                            ),
                        )
                    )
    return ExecutionVariables(execution_time=execution_time)


def _add_compute_queue_variables(
    model: gp.Model,
    problem: GurobiProblem,
    *,
    max_components: int,
    priority_count: int,
    compute_utilization_limit: float,
) -> ComputeQueueVariables:
    component_priority: dict[tuple[FlowId, int, WorkerId, int], gp.Var] = {}
    priority_class_utilization: dict[tuple[FlowId, int, WorkerId, int], gp.Var] = {}
    component_queue_wait: dict[tuple[FlowId, int], gp.Var] = {}
    priorities = range(1, priority_count + 1)

    for flow in problem.flows:
        flow_index = problem.flow_index[flow]
        for component in range(1, max_components + 1):
            component_queue_wait[flow, component] = model.addVar(
                lb=0.0,
                ub=problem.max_queue_wait_by_flow[flow],
                name=f"wex[{flow_index},{component}]",
            )
            for worker in problem.workers_by_flow[flow]:
                worker_index = problem.worker_index[worker]
                for priority in priorities:
                    component_priority[flow, component, worker, priority] = (
                        model.addVar(
                            vtype=GRB.BINARY,
                            name=(
                                f"r[{flow_index},{component},{worker_index},{priority}]"
                            ),
                        )
                    )
                    priority_class_utilization[flow, component, worker, priority] = (
                        model.addVar(
                            lb=0.0,
                            ub=compute_utilization_limit,
                            name=(
                                f"rho_bar[{flow_index},{component},{worker_index},"
                                f"{priority}]"
                            ),
                        )
                    )

    priority_used: dict[tuple[WorkerId, int], gp.Var] = {}
    lower_priority_utilization: dict[tuple[WorkerId, int], gp.Var] = {}
    compute_bin_selected: dict[tuple[WorkerId, int, int], gp.Var] = {}
    queue_wait_bin: dict[tuple[WorkerId, int, int], gp.Var] = {}
    queue_wait: dict[tuple[WorkerId, int], gp.Var] = {}
    weighted_queue_wait: dict[tuple[WorkerId, int], gp.Var] = {}
    remaining_compute_capacity: dict[tuple[WorkerId, int], gp.Var] = {}
    compute_bin_count = len(problem.compute_bin_limits) - 1

    for worker in problem.workers:
        worker_index = problem.worker_index[worker]
        max_wait = problem.max_queue_wait_by_worker[worker]
        for priority in priorities:
            priority_used[worker, priority] = model.addVar(
                lb=0.0,
                ub=1.0,
                name=f"n[{worker_index},{priority}]",
            )
            queue_wait[worker, priority] = model.addVar(
                lb=0.0,
                ub=max_wait,
                name=f"h[{worker_index},{priority}]",
            )
            remaining_compute_capacity[worker, priority] = model.addVar(
                lb=1 - compute_utilization_limit,
                ub=1.0,
                name=f"rho_remaining[{worker_index},{priority}]",
            )
            if priority < 2:
                continue
            lower_priority_utilization[worker, priority] = model.addVar(
                lb=0.0,
                ub=compute_utilization_limit,
                name=f"rho_lt[{worker_index},{priority}]",
            )
            weighted_queue_wait[worker, priority] = model.addVar(
                lb=0.0,
                ub=max_wait,
                name=f"h_weighted[{worker_index},{priority}]",
            )
            for bin_index in range(1, compute_bin_count + 1):
                compute_bin_selected[worker, priority, bin_index] = model.addVar(
                    vtype=GRB.BINARY,
                    name=f"g[{worker_index},{priority},{bin_index}]",
                )
                queue_wait_bin[worker, priority, bin_index] = model.addVar(
                    lb=0.0,
                    ub=max_wait,
                    name=f"h_bin[{worker_index},{priority},{bin_index}]",
                )

    return ComputeQueueVariables(
        component_priority=component_priority,
        priority_used=priority_used,
        priority_class_utilization=priority_class_utilization,
        lower_priority_utilization=lower_priority_utilization,
        compute_bin_selected=compute_bin_selected,
        queue_wait_bin=queue_wait_bin,
        queue_wait=queue_wait,
        weighted_queue_wait=weighted_queue_wait,
        remaining_compute_capacity=remaining_compute_capacity,
        component_queue_wait=component_queue_wait,
    )


def _add_network_variables(
    model: gp.Model,
    problem: GurobiProblem,
    context: OptimizationContext,
    requests_by_flow: OptimizationRequestsByFlow,
    *,
    max_components: int,
    network_utilization_limit: float,
) -> NetworkVariables:
    tensor_crosses_cut: dict[tuple[FlowId, ModelVariantId, TensorKey, int], gp.Var] = {}
    adjacent_component_workers: dict[
        tuple[FlowId, int, WorkerId, WorkerId], gp.Var
    ] = {}
    transmission_time: dict[tuple[FlowId, ModelVariantId, int, WorkerId], gp.Var] = {}
    component_transmission_response: dict[tuple[FlowId, int], gp.Var] = {}
    cuts = range(1, max_components)
    connections = context.network_topology.connections()

    for flow in problem.flows:
        flow_index = problem.flow_index[flow]
        max_response = float(
            requests_by_flow[flow].qos_requirements.max_mean_response_time
        )
        for model_variant in problem.models_by_flow[flow]:
            model_index = problem.model_index[flow, model_variant]
            static_profile = context.static_profiles_by_variant[model_variant]
            for tensor in tensor_routes(static_profile):
                tensor_index = problem.tensor_index[model_variant, tensor]
                for cut in cuts:
                    tensor_crosses_cut[flow, model_variant, tensor, cut] = model.addVar(
                        lb=0.0,
                        ub=1.0,
                        name=f"c[{flow_index},{model_index},{tensor_index},{cut}]",
                    )
            for cut in cuts:
                for worker in problem.workers_by_flow[flow]:
                    worker_index = problem.worker_index[worker]
                    transmission_time[flow, model_variant, cut, worker] = model.addVar(
                        lb=0.0,
                        ub=max_response,
                        name=f"ttx[{flow_index},{model_index},{cut},{worker_index}]",
                    )

        for cut in cuts:
            component_transmission_response[flow, cut] = model.addVar(
                lb=0.0,
                ub=max_response,
                name=f"rtx[{flow_index},{cut}]",
            )
            for source in problem.workers_by_flow[flow]:
                source_index = problem.worker_index[source]
                for target in problem.workers_by_flow[flow]:
                    if source != target and (source, target) not in connections:
                        continue
                    target_index = problem.worker_index[target]
                    adjacent_component_workers[flow, cut, source, target] = (
                        model.addVar(
                            vtype=GRB.BINARY,
                            name=(
                                f"q[{flow_index},{cut},{source_index},{target_index}]"
                            ),
                        )
                    )

    network_utilization: dict[WorkerId, gp.Var] = {}
    network_bin_selected: dict[tuple[WorkerId, int], gp.Var] = {}
    network_bin_count = len(problem.network_bin_limits) - 1
    for worker in problem.workers:
        worker_index = problem.worker_index[worker]
        network_utilization[worker] = model.addVar(
            lb=0.0,
            ub=network_utilization_limit,
            name=f"rho_tx[{worker_index}]",
        )
        for bin_index in range(1, network_bin_count + 1):
            network_bin_selected[worker, bin_index] = model.addVar(
                vtype=GRB.BINARY,
                name=f"g_tx[{worker_index},{bin_index}]",
            )

    return NetworkVariables(
        tensor_crosses_cut=tensor_crosses_cut,
        adjacent_component_workers=adjacent_component_workers,
        transmission_time=transmission_time,
        network_utilization=network_utilization,
        network_bin_selected=network_bin_selected,
        component_transmission_response=component_transmission_response,
    )


def _add_memory_variables(
    model: gp.Model,
    problem: GurobiProblem,
    context: OptimizationContext,
) -> MemoryVariables:
    return MemoryVariables(
        max_activation_bytes={
            worker: model.addVar(
                lb=0.0,
                ub=context.resource_profiles_by_worker[
                    worker
                ].ram_profile.available_bytes,
                name=f"q_activation[{problem.worker_index[worker]}]",
            )
            for worker in problem.workers
        }
    )


def _add_qos_variables(
    model: gp.Model,
    problem: GurobiProblem,
    requests_by_flow: OptimizationRequestsByFlow,
) -> QoSVariables:
    return QoSVariables(
        response_time={
            flow: model.addVar(
                lb=0.0,
                ub=float(
                    requests_by_flow[flow].qos_requirements.max_mean_response_time
                ),
                name=f"response[{problem.flow_index[flow]}]",
            )
            for flow in problem.flows
        }
    )


def add_gurobi_variables(
    model: gp.Model,
    problem: GurobiProblem,
    *,
    context: OptimizationContext,
    requests_by_flow: OptimizationRequestsByFlow,
    max_components: int,
    priority_count: int,
    compute_utilization_limit: float,
    network_utilization_limit: float,
) -> GurobiVariables:
    return GurobiVariables(
        assignment=_add_assignment_variables(
            model,
            problem,
            context,
            max_components=max_components,
        ),
        execution=_add_execution_variables(
            model,
            problem,
            max_components=max_components,
        ),
        compute_queue=_add_compute_queue_variables(
            model,
            problem,
            max_components=max_components,
            priority_count=priority_count,
            compute_utilization_limit=compute_utilization_limit,
        ),
        network=_add_network_variables(
            model,
            problem,
            context,
            requests_by_flow,
            max_components=max_components,
            network_utilization_limit=network_utilization_limit,
        ),
        memory=_add_memory_variables(model, problem, context),
        qos=_add_qos_variables(model, problem, requests_by_flow),
    )
