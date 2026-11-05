"""Compact variable creation, reusing unchanged group factories."""

import gurobipy as gp
from gurobipy import GRB

from controller.adapters.outbound.optimization.compact_memory import (
    add_compact_memory_variables,
)
from controller.adapters.outbound.optimization.compact_network_queue import (
    add_compact_network_variables,
)
from controller.adapters.outbound.optimization.gurobi_problem import GurobiProblem
from controller.adapters.outbound.optimization.gurobi_variables import (
    GurobiVariables,
    _add_assignment_variables,
    _add_compute_queue_variables,
    _add_execution_variables,
    _add_network_variables,
    _add_qos_variables,
)
from controller.domain.optimization.request.optimization_request import (
    OptimizationRequestsByFlow,
)
from controller.domain.optimization.solving.optimization_context import (
    OptimizationContext,
)


def add_compact_variables(
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
    variables = GurobiVariables(
        assignment=_add_assignment_variables(
            model, problem, context, max_components=max_components
        ),
        execution=_add_execution_variables(
            model, problem, max_components=max_components
        ),
        compute_queue=_add_compute_queue_variables(
            model,
            problem,
            max_components=max_components,
            priority_count=priority_count,
            compute_utilization_limit=compute_utilization_limit,
        ),
        network=add_compact_network_variables(
            model,
            problem,
            _add_network_variables(
                model,
                problem,
                context,
                requests_by_flow,
                max_components=max_components,
                network_utilization_limit=network_utilization_limit,
            ),
            context,
            max_components=max_components,
        ),
        memory=add_compact_memory_variables(
            model, problem, context, max_components=max_components
        ),
        qos=_add_qos_variables(model, problem, requests_by_flow),
    )
    # sum_p r[f,k,s,p] = e[f,k,s], with binary r and e in [0,1].
    for placement in variables.assignment.component_worker.values():
        placement.VType = GRB.CONTINUOUS
        placement.LB = 0
        placement.UB = 1
    # Sparse marginals determine q uniquely once placements are integral.
    for transition in variables.network.adjacent_component_workers.values():
        transition.VType = GRB.CONTINUOUS
        transition.LB = 0
        transition.UB = 1
    return variables
