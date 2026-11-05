"""Priority queues reuse conservation, prefix, discretization and SOC builders."""

import gurobipy as gp

from controller.adapters.outbound.optimization.gurobi_execution_queue_constraints import (
    add_execution_queue_constraints,
)
from controller.adapters.outbound.optimization.gurobi_problem import GurobiProblem
from controller.adapters.outbound.optimization.gurobi_variables import (
    AssignmentVariables,
    ComputeQueueVariables,
    ExecutionVariables,
)


def add_compact_execution_queue_constraints(
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
    # Reuse shared utilization conservation and one SOC per priority, then add
    # the compact physical upper bounds.
    add_execution_queue_constraints(
        model,
        problem,
        assignment,
        execution,
        compute_queue,
        max_components=max_components,
        priority_count=priority_count,
        compute_utilization_limit=compute_utilization_limit,
    )
    p = problem
    bounds = {}
    for worker in p.workers:
        worst = p.max_queue_wait_s_by_worker[worker]
        # Later priority-suffix strengthening disabled; retain original bounds.
        # relevant_flows = tuple(
        #     flow for flow in p.flows if worker in p.workers_by_flow[flow]
        # )
        for priority in range(1, priority_count + 1):
            # model.addConstr(
            #     gp.quicksum(
            #         compute_queue.priority_class_utilization[
            #             flow, component, worker, suffix_priority
            #         ]
            #         for suffix_priority in range(priority, priority_count + 1)
            #         for flow in relevant_flows
            #         for component in range(1, max_components + 1)
            #     )
            #     <= compute_utilization_limit
            #     * compute_queue.priority_used[worker, priority],
            #     name=f"compact_priority_suffix_utilization[{p.worker_index[worker]},{priority}]",
            # )
            upper = worst * (1 - compute_utilization_limit) if priority == 1 else worst
            bounds[worker, priority] = upper
            compute_queue.queue_wait_s[worker, priority].UB = upper
            if priority == 1:
                continue
            for index in range(1, len(p.compute_bin_limits)):
                bin_upper = (
                    worst
                    * (1 - compute_utilization_limit)
                    / (1 - p.compute_bin_limits[index])
                )
                wait = compute_queue.queue_wait_bin_s[worker, priority, index]
                wait.UB = bin_upper
                model.addConstr(
                    wait
                    <= bin_upper
                    * compute_queue.compute_bin_selected[worker, priority, index],
                    name=f"compact_wait_bin_bound[{p.worker_index[worker]},{priority},{index}]",
                )
    for flow in p.flows:
        for component in range(1, max_components + 1):
            wait = compute_queue.component_queue_wait_s[flow, component]
            terms = []
            for worker in p.workers_by_flow[flow]:
                for priority in range(1, priority_count + 1):
                    r = compute_queue.component_priority[
                        flow, component, worker, priority
                    ]
                    upper = bounds[worker, priority]
                    terms.append(upper * r)
                    model.addConstr(
                        wait
                        >= compute_queue.queue_wait_s[worker, priority]
                        - upper * (1 - r),
                        name=f"compact_wait_activation[{p.flow_index[flow]},{component},{p.worker_index[worker]},{priority}]",
                    )
            model.addConstr(
                wait <= gp.quicksum(terms),
                name=f"compact_component_wait_bound[{p.flow_index[flow]},{component}]",
            )
