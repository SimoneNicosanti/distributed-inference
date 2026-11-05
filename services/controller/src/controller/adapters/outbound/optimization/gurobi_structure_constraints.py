from collections.abc import Mapping

import gurobipy as gp

from controller.adapters.outbound.optimization.gurobi_expressions import (
    destination_crossing_expression,
)
from controller.adapters.outbound.optimization.gurobi_problem import (
    GurobiProblem,
)
from controller.adapters.outbound.optimization.gurobi_variables import (
    AssignmentVariables,
    NetworkVariables,
)
from controller.domain.optimization.request.flow_endpoint_placement import (
    FixedPlacement,
)
from controller.domain.optimization.request.optimization_request import (
    OptimizationRequestsByFlow,
)
from controller.domain.optimization.solving.optimization_context import (
    OptimizationContext,
)
from controller.domain.profile.model_variant_execution_profile import (
    LayerExecutionProfile,
)
from controller.domain.profile.model_variant_static_profile import (
    ModelVariantStaticProfile,
)
from controller.domain.profile.model_variant_topology import LayerDescription
from controller.domain.profile.network_topology import (
    NetworkTopology,
)
from shared.flow.flow import FlowId
from shared.model.keys import LayerKey
from shared.model.model_variant import ModelVariantId
from shared.service.service import WorkerId


def add_assignment_constraints(
    model: gp.Model,
    problem: GurobiProblem,
    assignment: AssignmentVariables,
    network: NetworkVariables,
    *,
    context: OptimizationContext,
    requests_by_flow: OptimizationRequestsByFlow,
    max_components: int,
) -> None:
    p = problem
    components = range(1, max_components + 1)
    cuts = range(1, max_components)

    for flow in p.flows:
        flow_index = p.flow_index[flow]
        _one_model_per_flow_constraint(model, assignment, p, flow, flow_index)
        for component in components:
            _component_is_active_if_has_worker(
                model, assignment, p, flow, flow_index, component
            )
            _component_is_active_if_has_at_least_one_layer(
                model, assignment, context, p, flow, flow_index, component
            )
            _component_is_last(model, assignment, flow, flow_index, component)

            _component_sequence_is_prefix(
                model, assignment, flow, flow_index, component
            )

        _endpoint_placement_constraints(
            model, assignment, requests_by_flow, components, flow, flow_index
        )
        for cut in cuts:
            _successive_components_use_different_workers(
                model, assignment, p, flow, flow_index, cut
            )

        for model_variant in p.models_by_flow[flow]:
            model_index = p.model_index[flow, model_variant]
            static_profile = context.static_profiles_by_variant[model_variant]
            _add_model_assignment_constraints(
                model,
                p,
                assignment,
                network,
                flow=flow,
                model_variant=model_variant,
                static_profile=static_profile,
                flow_index=flow_index,
                model_index=model_index,
                components=components,
                cuts=cuts,
            )


def _endpoint_placement_constraints(
    model: gp.Model,
    assignment: AssignmentVariables,
    requests_by_flow: OptimizationRequestsByFlow,
    components: range,
    flow: FlowId,
    flow_index: int,
) -> None:
    placement = requests_by_flow[flow].flow_endpoint_placement
    if isinstance(placement.ingress_placement, FixedPlacement):
        ingress = placement.ingress_placement.worker_id
        model.addConstr(
            assignment.component_worker[flow, 1, ingress] == 1,
            name=f"first_on_device[{flow_index}]",
        )
    if isinstance(placement.sink_placement, FixedPlacement):
        sink = placement.sink_placement.worker_id
        for component in components:
            model.addConstr(
                assignment.component_worker[flow, component, sink]
                >= assignment.component_last[flow, component],
                name=f"last_on_device[{flow_index},{component}]",
            )


def _successive_components_use_different_workers(
    model: gp.Model,
    assignment: AssignmentVariables,
    p: GurobiProblem,
    flow: FlowId,
    flow_index: int,
    cut: int,
) -> None:
    for worker in p.workers_by_flow[flow]:
        model.addConstr(
            assignment.component_worker[flow, cut, worker]
            + assignment.component_worker[flow, cut + 1, worker]
            <= 1,
            name=(
                f"successive_components_different_workers["
                f"{flow_index},{cut},{p.worker_index[worker]}]"
            ),
        )


def _component_sequence_is_prefix(
    model: gp.Model,
    assignment: AssignmentVariables,
    flow: FlowId,
    flow_index: int,
    component: int,
) -> None:
    model.addConstr(
        assignment.component_active_at(flow, component + 1)
        <= assignment.component_active_at(flow, component),
        name=f"active_prefix[{flow_index},{component}]",
    )


def _component_is_last(
    model: gp.Model,
    assignment: AssignmentVariables,
    flow: FlowId,
    flow_index: int,
    component: int,
) -> None:
    model.addConstr(
        assignment.component_last[flow, component]
        == assignment.component_active_at(flow, component)
        - assignment.component_active_at(flow, component + 1),
        name=f"last_component[{flow_index},{component}]",
    )


def _component_is_active_if_has_at_least_one_layer(
    model: gp.Model,
    assignment: AssignmentVariables,
    context: OptimizationContext,
    p: GurobiProblem,
    flow: FlowId,
    flow_index: int,
    component: int,
) -> None:
    layer_assignments: list[gp.Var] = []
    for model_variant in p.models_by_flow[flow]:
        model_index = p.model_index[flow, model_variant]
        static_profile = context.static_profiles_by_variant[model_variant]
        for layer in static_profile.topology.layers_in_topological_order():
            layer_index = p.layer_index[model_variant, layer]
            layer_assignment = assignment.layer_component[
                flow, model_variant, layer, component
            ]
            layer_assignments.append(layer_assignment)
            model.addConstr(
                assignment.component_active[flow, component] >= layer_assignment,
                name=(
                    f"active_if_layer[{flow_index},{model_index},"
                    f"{layer_index},{component}]"
                ),
            )
    model.addConstr(
        assignment.component_active[flow, component] <= gp.quicksum(layer_assignments),
        name=f"inactive_without_layer[{flow_index},{component}]",
    )


def _component_is_active_if_has_worker(
    model: gp.Model,
    assignment: AssignmentVariables,
    p: GurobiProblem,
    flow: FlowId,
    flow_index: int,
    component: int,
) -> None:
    model.addConstr(
        assignment.component_active[flow, component]
        == gp.quicksum(
            assignment.component_worker[flow, component, worker]
            for worker in p.workers_by_flow[flow]
        ),
        name=f"active_worker[{flow_index},{component}]",
    )


def _one_model_per_flow_constraint(
    model: gp.Model,
    assignment: AssignmentVariables,
    p: GurobiProblem,
    flow: FlowId,
    flow_index: int,
) -> None:
    model.addConstr(
        gp.quicksum(
            assignment.model_selected[flow, model_variant]
            for model_variant in p.models_by_flow[flow]
        )
        == 1,
        name=f"choose_model[{flow_index}]",
    )


def _add_model_assignment_constraints(
    model: gp.Model,
    p: GurobiProblem,
    assignment: AssignmentVariables,
    network: NetworkVariables,
    *,
    flow: FlowId,
    model_variant: ModelVariantId,
    static_profile: ModelVariantStaticProfile,
    flow_index: int,
    model_index: int,
    components: range,
    cuts: range,
) -> None:
    layer_descriptions = static_profile.topology.layers()
    _layer_assigned_to_component_if_model_selected(
        model,
        p,
        assignment,
        flow,
        model_variant,
        static_profile,
        flow_index,
        model_index,
        components,
    )
    _input_layers_on_first_component(
        model,
        p,
        assignment,
        flow,
        model_variant,
        flow_index,
        model_index,
        layer_descriptions,
    )
    _output_layers_on_last_component(
        model,
        p,
        assignment,
        flow,
        model_variant,
        flow_index,
        model_index,
        components,
        layer_descriptions,
    )
    _layer_assignments_respect_precedence(
        model,
        p,
        assignment,
        flow,
        model_variant,
        static_profile,
        flow_index,
        model_index,
        components,
    )
    _tensor_crossing_cut(
        model,
        p,
        assignment,
        network,
        flow=flow,
        model_variant=model_variant,
        static_profile=static_profile,
        flow_index=flow_index,
        model_index=model_index,
        cuts=cuts,
    )


## TODO: Double check this part
def _tensor_crossing_cut(
    model: gp.Model,
    p: GurobiProblem,
    assignment: AssignmentVariables,
    network: NetworkVariables,
    *,
    flow: FlowId,
    model_variant: ModelVariantId,
    static_profile: ModelVariantStaticProfile,
    flow_index: int,
    model_index: int,
    cuts: range,
) -> None:
    for tensor, route in static_profile.topology.tensor_routes().items():
        if len(route.maximal_destinations) <= 1:
            continue
        tensor_index = p.tensor_index[model_variant, tensor]
        for cut in cuts:
            crosses = network.tensor_crossing_auxiliary[
                flow, model_variant, tensor, cut
            ]
            destination_crossings = tuple(
                (
                    destination,
                    destination_crossing_expression(
                        assignment,
                        flow=flow,
                        model_variant=model_variant,
                        source=route.source,
                        destination=destination,
                        cut=cut,
                    ),
                )
                for destination in route.maximal_destinations
            )
            for destination, destination_crossing in destination_crossings:
                destination_index = p.layer_index[model_variant, destination]
                model.addConstr(
                    crosses >= destination_crossing,
                    name=(
                        f"cut_lower[{flow_index},{model_index},{tensor_index},"
                        f"{cut},{destination_index}]"
                    ),
                )
            model.addConstr(
                crosses
                <= gp.quicksum(crossing for _, crossing in destination_crossings),
                name=f"cut_upper[{flow_index},{model_index},{tensor_index},{cut}]",
            )


def _layer_assignments_respect_precedence(
    model: gp.Model,
    p: GurobiProblem,
    assignment: AssignmentVariables,
    flow: FlowId,
    model_variant: ModelVariantId,
    static_profile: ModelVariantStaticProfile,
    flow_index: int,
    model_index: int,
    components: range,
) -> None:
    for source, target in static_profile.topology.edges():
        source_index = p.layer_index[model_variant, source]
        target_index = p.layer_index[model_variant, target]
        for component in components:
            model.addConstr(
                gp.quicksum(
                    assignment.layer_component[flow, model_variant, target, previous]
                    for previous in range(1, component + 1)
                )
                <= gp.quicksum(
                    assignment.layer_component[flow, model_variant, source, previous]
                    for previous in range(1, component + 1)
                ),
                name=(
                    f"precedence[{flow_index},{model_index},{source_index},"
                    f"{target_index},{component}]"
                ),
            )


def _output_layers_on_last_component(
    model: gp.Model,
    p: GurobiProblem,
    assignment: AssignmentVariables,
    flow: FlowId,
    model_variant: ModelVariantId,
    flow_index: int,
    model_index: int,
    components: range,
    layer_descriptions: Mapping[LayerKey, LayerDescription],
) -> None:
    for layer, description in layer_descriptions.items():
        if not description.is_output:
            continue
        layer_index = p.layer_index[model_variant, layer]
        for component in components:
            model.addConstr(
                assignment.layer_component[flow, model_variant, layer, component]
                >= assignment.model_selected[flow, model_variant]
                + assignment.component_last[flow, component]
                - 1,
                name=(
                    f"output_last[{flow_index},{model_index},{layer_index},{component}]"
                ),
            )


def _input_layers_on_first_component(
    model: gp.Model,
    p: GurobiProblem,
    assignment: AssignmentVariables,
    flow: FlowId,
    model_variant: ModelVariantId,
    flow_index: int,
    model_index: int,
    layer_descriptions: Mapping[LayerKey, LayerDescription],
) -> None:
    for layer, description in layer_descriptions.items():
        if not description.is_input:
            continue
        model.addConstr(
            assignment.layer_component[flow, model_variant, layer, 1]
            == assignment.model_selected[flow, model_variant],
            name=(
                f"input_first[{flow_index},{model_index},"
                f"{p.layer_index[model_variant, layer]}]"
            ),
        )


def _layer_assigned_to_component_if_model_selected(
    model: gp.Model,
    p: GurobiProblem,
    assignment: AssignmentVariables,
    flow: FlowId,
    model_variant: ModelVariantId,
    static_profile: ModelVariantStaticProfile,
    flow_index: int,
    model_index: int,
    components: range,
) -> None:
    for layer in static_profile.topology.layers_in_topological_order():
        layer_index = p.layer_index[model_variant, layer]
        model.addConstr(
            gp.quicksum(
                assignment.layer_component[flow, model_variant, layer, component]
                for component in components
            )
            == assignment.model_selected[flow, model_variant],
            name=f"assign_layer[{flow_index},{model_index},{layer_index}]",
        )


def _minimum_execution_time_by_layer(
    layers: tuple[LayerKey, ...],
    profiles_by_worker: dict[WorkerId, Mapping[LayerKey, LayerExecutionProfile] | None],
) -> dict[LayerKey, float]:
    result: dict[LayerKey, float] = {}
    for layer in layers:
        compatible_times = tuple(
            float(profiles[layer].execution_time_s)
            for profiles in profiles_by_worker.values()
            if profiles is not None and layer in profiles
        )
        if compatible_times:
            result[layer] = min(compatible_times)
    return result


def _add_transition_marginal_constraints(
    model: gp.Model,
    problem: GurobiProblem,
    assignment: AssignmentVariables,
    network: NetworkVariables,
    *,
    topology: NetworkTopology,
    flow: FlowId,
    cut: int,
) -> None:
    flow_index = problem.flow_index[flow]
    workers = problem.workers_by_flow[flow]

    for source in workers:
        successors = topology.successors(source)
        outgoing = tuple(
            network.adjacent_component_workers[flow, cut, source, target]
            for target in workers
            if source == target or target in successors
        )
        model.addConstr(
            gp.quicksum(outgoing) <= assignment.component_worker[flow, cut, source],
            name=(f"q_outgoing[{flow_index},{cut},{problem.worker_index[source]}]"),
        )

    for target in workers:
        predecessors = topology.predecessors(target)
        incoming = (
            network.adjacent_component_workers[flow, cut, source, target]
            for source in workers
            if source == target or source in predecessors
        )
        model.addConstr(
            gp.quicksum(incoming) == assignment.component_worker[flow, cut + 1, target],
            name=f"q_incoming[{flow_index},{cut},{problem.worker_index[target]}]",
        )
