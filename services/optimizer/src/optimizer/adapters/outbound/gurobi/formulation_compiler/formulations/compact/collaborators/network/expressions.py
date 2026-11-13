import gurobipy as gp

from optimizer.adapters.outbound.gurobi.formulation_compiler.formulations.compact.collaborators.assignment.assignment_variables import (
    AssignmentVariables,
)
from optimizer.adapters.outbound.gurobi.formulation_compiler.formulations.compact.collaborators.network.network_variables import (
    NetworkVariables,
)
from optimizer.adapters.outbound.gurobi.gurobi_units import (
    bytes_to_megabytes,
)
from optimizer.domain.formulation.compact.misocp.misocp_compact_formulation import (
    MisocpCompactFormulation,
    TensorRoute,
)
from optimizer.domain.problem.compact.compact_optimization_problem import (
    ModelConfigurationKey,
)
from shared.flow.flow import FlowId
from shared.model.keys import LayerKey, TensorKey

type GurobiAffineTerm = gp.Var | gp.LinExpr


def destination_crossing_expression(
    assignment: AssignmentVariables,
    *,
    flow: FlowId,
    model_variant: ModelConfigurationKey,
    source: LayerKey,
    destination: LayerKey,
    cut: int,
) -> gp.LinExpr:
    return gp.quicksum(
        assignment.layer_component[flow, model_variant, source, component]
        - assignment.layer_component[flow, model_variant, destination, component]
        for component in range(1, cut + 1)
    )


def tensor_crossing_term(
    assignment: AssignmentVariables,
    network: NetworkVariables,
    *,
    flow: FlowId,
    model_variant: ModelConfigurationKey,
    tensor: TensorKey,
    route: TensorRoute,
    cut: int,
) -> GurobiAffineTerm:
    if not route.maximal_destinations:
        return gp.LinExpr(0.0)
    if len(route.maximal_destinations) == 1:
        return destination_crossing_expression(
            assignment,
            flow=flow,
            model_variant=model_variant,
            source=route.source,
            destination=route.maximal_destinations[0],
            cut=cut,
        )
    return network.tensor_crossing_auxiliary[flow, model_variant, tensor, cut]


def transmitted_data_size_mb_expression(
    assignment: AssignmentVariables,
    network: NetworkVariables,
    formulation: MisocpCompactFormulation,
    *,
    flow: FlowId,
    model_variant: ModelConfigurationKey,
    cut: int,
) -> gp.LinExpr:
    return gp.quicksum(
        bytes_to_megabytes(
            formulation.problem.configurations[model_variant].tensor_bytes[tensor]
        )
        * tensor_crossing_term(
            assignment,
            network,
            flow=flow,
            model_variant=model_variant,
            tensor=tensor,
            route=route,
            cut=cut,
        )
        for tensor, route in formulation.tensor_routes_by_variant[
            model_variant.model_variant_id
        ].items()
    )
