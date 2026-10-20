import gurobipy as gp

from controller.adapters.outbound.optimization.gurobi_profile_data import (
    tensor_size_bytes,
)
from controller.adapters.outbound.optimization.gurobi_variables import (
    AssignmentVariables,
    NetworkVariables,
)
from controller.domain.profile.model_variant_static_profile import (
    ModelVariantStaticProfile,
)
from controller.domain.profile.model_variant_topology import TensorRoute
from shared.flow.flow import FlowId
from shared.model.keys import LayerKey, TensorKey
from shared.model.model_variant import ModelVariantId

type GurobiAffineTerm = gp.Var | gp.LinExpr


def destination_crossing_expression(
    assignment: AssignmentVariables,
    *,
    flow: FlowId,
    model_variant: ModelVariantId,
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
    model_variant: ModelVariantId,
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


def transmitted_data_size_expression(
    assignment: AssignmentVariables,
    network: NetworkVariables,
    profile: ModelVariantStaticProfile,
    *,
    flow: FlowId,
    model_variant: ModelVariantId,
    cut: int,
) -> gp.LinExpr:
    return gp.quicksum(
        tensor_size_bytes(profile, tensor)
        * tensor_crossing_term(
            assignment,
            network,
            flow=flow,
            model_variant=model_variant,
            tensor=tensor,
            route=route,
            cut=cut,
        )
        for tensor, route in profile.topology.tensor_routes().items()
    )
