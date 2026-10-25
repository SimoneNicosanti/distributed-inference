import gurobipy as gp

from controller.adapters.outbound.optimization.gurobi_problem import (
    GurobiProblem,
)
from controller.adapters.outbound.optimization.gurobi_variables import (
    AssignmentVariables,
    ComputeQueueVariables,
    ExecutionVariables,
)
from shared.flow.flow import FlowId
from shared.service.service import WorkerId


def add_execution_queue_constraints(
    model: gp.Model,
    problem: GurobiProblem,
    assignment: AssignmentVariables,
    execution: ExecutionVariables,
    compute_queue: ComputeQueueVariables,
    *,
    max_components: int,
    priority_count: int,
    compute_utilization_limit: float,
) -> None:
    p = problem
    priorities = range(1, priority_count + 1)
    compute_bin_count = len(p.compute_bin_limits) - 1
    class_utilization = {
        (flow, component, worker): gp.quicksum(
            p.service_arrival_rate_per_s[flow, model_variant]
            * execution.execution_time_s[flow, model_variant, component, worker]
            for model_variant in p.models_by_flow[flow]
        )
        for flow in p.flows
        for component in range(1, max_components + 1)
        for worker in p.workers_by_flow[flow]
    }

    for flow in p.flows:
        flow_index = p.flow_index[flow]
        for component in range(1, max_components + 1):
            for worker in p.workers_by_flow[flow]:
                worker_index = p.worker_index[worker]
                _worker_priority_assigned_to_executed_components(
                    model,
                    assignment,
                    compute_queue,
                    priorities,
                    flow,
                    flow_index,
                    component,
                    worker,
                    worker_index,
                )
                _define_component_utilization_for_selected_priority(
                    model,
                    compute_queue,
                    compute_utilization_limit,
                    priorities,
                    class_utilization,
                    flow,
                    flow_index,
                    component,
                    worker,
                    worker_index,
                )

    for worker in p.workers:
        worker_index = p.worker_index[worker]
        relevant_flows = tuple(
            flow for flow in p.flows if worker in p.workers_by_flow[flow]
        )
        _add_priority_usage_constraints(
            model,
            p,
            compute_queue,
            worker=worker,
            worker_index=worker_index,
            relevant_flows=relevant_flows,
            max_components=max_components,
            priority_count=priority_count,
        )
        _add_worker_total_utilization_constraint(
            model,
            max_components,
            compute_utilization_limit,
            class_utilization,
            worker,
            worker_index,
            relevant_flows,
        )

        second_moment = _compute_second_moment(
            execution, max_components, p, worker, relevant_flows
        )
        max_wait = p.max_queue_wait_s_by_worker[worker]
        for priority in priorities:
            higher_priority_rho = _compute_higher_priority_rho(
                compute_queue, max_components, worker, relevant_flows, priority
            )
            higher_equal_priority_rho = _compute_higher_equal_priority_rho(
                compute_queue, max_components, worker, relevant_flows, priority
            )
            _define_one_minus_rho_for_higher_equal_priority(
                model,
                compute_queue,
                worker,
                worker_index,
                priority,
                higher_equal_priority_rho,
            )

            weighted_queue_wait = gp.quicksum(
                (compute_queue.queue_wait_s[worker, priority],)
            )
            if priority >= 2:
                model.addConstr(
                    compute_queue.higher_priority_utilization[worker, priority]
                    == higher_priority_rho,
                    name=f"rho_lt_value[{worker_index},{priority}]",
                )
                _add_compute_discretization_constraints(
                    model,
                    p,
                    compute_queue,
                    worker=worker,
                    worker_index=worker_index,
                    priority=priority,
                    bin_count=compute_bin_count,
                    max_wait=max_wait,
                )
                weighted_queue_wait = gp.quicksum(
                    (compute_queue.weighted_queue_wait_s[worker, priority],)
                )

            # H <= 2 * h_hat * (1-rho_le): rotated SOC, all factors nonnegative.
            model.addQConstr(
                second_moment
                <= 2
                * weighted_queue_wait
                * compute_queue.remaining_compute_capacity[worker, priority],
                name=f"queue_soc[{worker_index},{priority}]",
            )

    for flow in p.flows:
        flow_index = p.flow_index[flow]
        max_flow_wait = p.max_queue_wait_s_by_flow[flow]
        for component in range(1, max_components + 1):
            _add_component_execution_wait_activation_constraints(
                model,
                assignment,
                compute_queue,
                p,
                priorities,
                flow,
                flow_index,
                component,
                max_flow_wait,
            )


def _add_component_execution_wait_activation_constraints(
    model: gp.Model,
    assignment: AssignmentVariables,
    compute_queue: ComputeQueueVariables,
    p: GurobiProblem,
    priorities: range,
    flow: FlowId,
    flow_index: int,
    component: int,
    max_flow_wait: float,
) -> None:
    model.addConstr(
        compute_queue.component_queue_wait_s[flow, component]
        <= max_flow_wait * assignment.component_active[flow, component],
        name=f"inactive_queue_wait[{flow_index},{component}]",
    )
    for worker in p.workers_by_flow[flow]:
        worker_index = p.worker_index[worker]
        worker_max_wait = p.max_queue_wait_s_by_worker[worker]
        for priority in priorities:
            model.addConstr(
                compute_queue.component_queue_wait_s[flow, component]
                >= compute_queue.queue_wait_s[worker, priority]
                - worker_max_wait
                * (
                    1
                    - compute_queue.component_priority[
                        flow, component, worker, priority
                    ]
                ),
                name=(
                    f"component_queue_wait[{flow_index},{component},"
                    f"{worker_index},{priority}]"
                ),
            )


def _define_one_minus_rho_for_higher_equal_priority(
    model: gp.Model,
    compute_queue: ComputeQueueVariables,
    worker: WorkerId,
    worker_index: int,
    priority: int,
    higher_equal_priority_rho: gp.LinExpr,
) -> None:
    model.addConstr(
        compute_queue.remaining_compute_capacity[worker, priority]
        == 1 - higher_equal_priority_rho,
        name=f"remaining_compute[{worker_index},{priority}]",
    )


def _compute_higher_equal_priority_rho(
    compute_queue: ComputeQueueVariables,
    max_components: int,
    worker: WorkerId,
    relevant_flows: tuple[FlowId, ...],
    priority: int,
) -> gp.LinExpr:
    return gp.quicksum(
        compute_queue.priority_class_utilization[
            flow, component, worker, higher_priority
        ]
        for higher_priority in range(1, priority + 1)
        for flow in relevant_flows
        for component in range(1, max_components + 1)
    )


def _compute_higher_priority_rho(
    compute_queue: ComputeQueueVariables,
    max_components: int,
    worker: WorkerId,
    relevant_flows: tuple[FlowId, ...],
    priority: int,
) -> gp.LinExpr:
    return gp.quicksum(
        compute_queue.priority_class_utilization[
            flow, component, worker, higher_priority
        ]
        for higher_priority in range(1, priority)
        for flow in relevant_flows
        for component in range(1, max_components + 1)
    )


def _compute_second_moment(
    execution: ExecutionVariables,
    max_components: int,
    p: GurobiProblem,
    worker: WorkerId,
    relevant_flows: tuple[FlowId, ...],
) -> gp.QuadExpr:
    return gp.quicksum(
        p.service_arrival_rate_per_s[flow, model_variant]
        * execution.execution_time_s[flow, model_variant, component, worker]
        * execution.execution_time_s[flow, model_variant, component, worker]
        for flow in relevant_flows
        for model_variant in p.models_by_flow[flow]
        for component in range(1, max_components + 1)
    )


def _add_worker_total_utilization_constraint(
    model: gp.Model,
    max_components: int,
    compute_utilization_limit: float,
    class_utilization: dict[tuple[FlowId, int, WorkerId], gp.LinExpr],
    worker: WorkerId,
    worker_index: int,
    relevant_flows: tuple[FlowId, ...],
) -> None:
    compute_utilization = gp.quicksum(
        class_utilization[flow, component, worker]
        for flow in relevant_flows
        for component in range(1, max_components + 1)
    )
    model.addConstr(
        compute_utilization <= compute_utilization_limit,
        name=f"compute_stability[{worker_index}]",
    )


def _define_component_utilization_for_selected_priority(
    model: gp.Model,
    compute_queue: ComputeQueueVariables,
    compute_utilization_limit: float,
    priorities: range,
    class_utilization: dict[tuple[FlowId, int, WorkerId], gp.LinExpr],
    flow: FlowId,
    flow_index: int,
    component: int,
    worker: WorkerId,
    worker_index: int,
) -> None:
    for priority in priorities:
        model.addConstr(
            compute_queue.priority_class_utilization[flow, component, worker, priority]
            <= compute_utilization_limit
            * compute_queue.component_priority[flow, component, worker, priority],
            name=(f"rho_bar_bound[{flow_index},{component},{worker_index},{priority}]"),
        )
    model.addConstr(
        gp.quicksum(
            compute_queue.priority_class_utilization[flow, component, worker, priority]
            for priority in priorities
        )
        == class_utilization[flow, component, worker],
        name=f"rho_bar_sum[{flow_index},{component},{worker_index}]",
    )


def _worker_priority_assigned_to_executed_components(
    model: gp.Model,
    assignment: AssignmentVariables,
    compute_queue: ComputeQueueVariables,
    priorities: range,
    flow: FlowId,
    flow_index: int,
    component: int,
    worker: WorkerId,
    worker_index: int,
) -> None:
    model.addConstr(
        gp.quicksum(
            compute_queue.component_priority[flow, component, worker, priority]
            for priority in priorities
        )
        == assignment.component_worker[flow, component, worker],
        name=(f"priority_assignment[{flow_index},{component},{worker_index}]"),
    )


def _add_priority_usage_constraints(
    model: gp.Model,
    p: GurobiProblem,
    compute_queue: ComputeQueueVariables,
    *,
    worker: WorkerId,
    worker_index: int,
    relevant_flows: tuple[FlowId, ...],
    max_components: int,
    priority_count: int,
) -> None:
    for priority in range(1, priority_count + 1):
        assignments = tuple(
            compute_queue.component_priority[flow, component, worker, priority]
            for flow in relevant_flows
            for component in range(1, max_components + 1)
        )
        model.addConstr(
            compute_queue.priority_used[worker, priority] <= gp.quicksum(assignments),
            name=f"priority_used_upper[{worker_index},{priority}]",
        )
        for index, assignment in enumerate(assignments):
            model.addConstr(
                compute_queue.priority_used[worker, priority] >= assignment,
                name=f"priority_used_lower[{worker_index},{priority},{index}]",
            )
    for priority in range(1, priority_count):
        model.addConstr(
            compute_queue.priority_used[worker, priority]
            >= compute_queue.priority_used[worker, priority + 1],
            name=f"priority_prefix[{worker_index},{priority}]",
        )


def _add_compute_discretization_constraints(
    model: gp.Model,
    p: GurobiProblem,
    compute_queue: ComputeQueueVariables,
    *,
    worker: WorkerId,
    worker_index: int,
    priority: int,
    bin_count: int,
    max_wait: float,
) -> None:
    model.addConstr(
        gp.quicksum(
            compute_queue.compute_bin_selected[worker, priority, bin_index]
            for bin_index in range(1, bin_count + 1)
        )
        == 1,
        name=f"choose_compute_bin[{worker_index},{priority}]",
    )
    model.addConstr(
        gp.quicksum(
            p.compute_bin_limits[bin_index - 1]
            * compute_queue.compute_bin_selected[worker, priority, bin_index]
            for bin_index in range(1, bin_count + 1)
        )
        <= compute_queue.higher_priority_utilization[worker, priority],
        name=f"compute_bin_lower[{worker_index},{priority}]",
    )
    model.addConstr(
        compute_queue.higher_priority_utilization[worker, priority]
        <= gp.quicksum(
            p.compute_bin_limits[bin_index]
            * compute_queue.compute_bin_selected[worker, priority, bin_index]
            for bin_index in range(1, bin_count + 1)
        ),
        name=f"compute_bin_upper[{worker_index},{priority}]",
    )
    for bin_index in range(1, bin_count + 1):
        model.addConstr(
            compute_queue.queue_wait_bin_s[worker, priority, bin_index]
            <= max_wait
            * compute_queue.compute_bin_selected[worker, priority, bin_index],
            name=f"queue_wait_bin_active[{worker_index},{priority},{bin_index}]",
        )
    model.addConstr(
        compute_queue.queue_wait_s[worker, priority]
        == gp.quicksum(
            compute_queue.queue_wait_bin_s[worker, priority, bin_index]
            for bin_index in range(1, bin_count + 1)
        ),
        name=f"queue_wait_sum[{worker_index},{priority}]",
    )
    model.addConstr(
        compute_queue.weighted_queue_wait_s[worker, priority]
        == gp.quicksum(
            (1 - p.compute_bin_limits[bin_index])
            * compute_queue.queue_wait_bin_s[worker, priority, bin_index]
            for bin_index in range(1, bin_count + 1)
        ),
        name=f"weighted_queue_wait[{worker_index},{priority}]",
    )
