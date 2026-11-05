"""Compact assignment strengthening over the unchanged shared structure."""

import gurobipy as gp

from controller.adapters.outbound.optimization.gurobi_problem import GurobiProblem
from controller.adapters.outbound.optimization.gurobi_structure_constraints import (
    add_assignment_constraints,
)
from controller.adapters.outbound.optimization.gurobi_variables import (
    AssignmentVariables,
    NetworkVariables,
)
from controller.domain.optimization.request.optimization_request import (
    OptimizationRequestsByFlow,
)
from controller.domain.optimization.solving.optimization_context import (
    OptimizationContext,
)


def add_compact_assignment_constraints(
    model: gp.Model,
    problem: GurobiProblem,
    assignment: AssignmentVariables,
    network: NetworkVariables,
    *,
    context: OptimizationContext,
    requests_by_flow: OptimizationRequestsByFlow,
    max_components: int,
) -> None:
    add_assignment_constraints(
        model,
        problem,
        assignment,
        network,
        context=context,
        requests_by_flow=requests_by_flow,
        max_components=max_components,
    )
    for flow in problem.flows:
        outputs = {}
        for variant in problem.models_by_flow[flow]:
            topology = context.static_profiles_by_variant[variant].topology
            descriptions = topology.layers()
            # Preprocessing guarantees at least one output per candidate model.
            outputs[variant] = next(
                layer
                for layer in topology.layers_in_topological_order()
                if descriptions[layer].is_output
            )
        for component in range(1, max_components + 1):
            model.addConstr(
                gp.quicksum(
                    assignment.layer_component[flow, variant, output, component]
                    for variant, output in outputs.items()
                )
                == assignment.component_last[flow, component],
                name=f"compact_output_last[{problem.flow_index[flow]},{component}]",
            )
