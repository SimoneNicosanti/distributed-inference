from collections.abc import Mapping

import gurobipy as gp

from controller.adapters.outbound.optimization.gurobi.model_builder.formulations.compact.collaborators.assignment.assignment_variables import (
    AssignmentVariables,
)
from controller.domain.optimization.solving.formulation.compact.compact_formulation_data import (
    CompactFormulationData,
)
from controller.domain.optimization.request.flow_endpoint_placement import (
    FixedPlacement,
)
from controller.domain.profile.model_variant_static_profile import (
    ModelVariantStaticProfile,
)
from controller.domain.profile.model_variant_topology import LayerDescription
from shared.flow.flow import FlowId
from shared.model.keys import LayerKey
from shared.model.model_variant import ModelVariantId


def add_assignment_constraints(
    model: gp.Model, data: CompactFormulationData, *, assignment: AssignmentVariables
) -> None:
    problem = data
    max_components = data.max_components
    _add_assignment_constraints(
        model, problem, assignment, max_components=max_components
    )
    for flow in problem.flows:
        outputs = {}
        for variant in problem.models_by_flow[flow]:
            topology = problem.static_profiles_by_variant[variant].topology
            descriptions = topology.layers()
            outputs[variant] = next(
                layer
                for layer in topology.layers_in_topological_order()
                if descriptions[layer].is_output
            )
        for component in range(1, max_components + 1):
            model.addConstr(
                gp.quicksum(
                    (
                        assignment.layer_component[flow, variant, output, component]
                        for variant, output in outputs.items()
                    )
                )
                == assignment.component_last[flow, component],
                name=f"compact_output_last[{problem.flow_index[flow]},{component}]",
            )


def _add_assignment_constraints(
    model: gp.Model,
    problem: CompactFormulationData,
    assignment: AssignmentVariables,
    *,
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
                model, assignment, p, flow, flow_index, component
            )
            _component_is_last(model, assignment, flow, flow_index, component)
            _component_sequence_is_prefix(
                model, assignment, flow, flow_index, component
            )
        _endpoint_placement_constraints(
            model, assignment, problem, components, flow, flow_index
        )
        for cut in cuts:
            _successive_components_use_different_workers(
                model, assignment, p, flow, flow_index, cut
            )
        for model_variant in p.models_by_flow[flow]:
            model_index = p.model_index[flow, model_variant]
            static_profile = problem.static_profiles_by_variant[model_variant]
            _add_model_assignment_constraints(
                model,
                p,
                assignment,
                flow=flow,
                model_variant=model_variant,
                static_profile=static_profile,
                flow_index=flow_index,
                model_index=model_index,
                components=components,
            )


def _endpoint_placement_constraints(
    model: gp.Model,
    assignment: AssignmentVariables,
    problem: CompactFormulationData,
    components: range,
    flow: FlowId,
    flow_index: int,
) -> None:
    placement = problem.endpoint_placement_by_flow[flow]
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
    p: CompactFormulationData,
    flow: FlowId,
    flow_index: int,
    cut: int,
) -> None:
    for worker in p.workers_by_flow[flow]:
        model.addConstr(
            assignment.component_worker[flow, cut, worker]
            + assignment.component_worker[flow, cut + 1, worker]
            <= 1,
            name=f"successive_components_different_workers[{flow_index},{cut},{p.worker_index[worker]}]",
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
    p: CompactFormulationData,
    flow: FlowId,
    flow_index: int,
    component: int,
) -> None:
    layer_assignments: list[gp.Var] = []
    for model_variant in p.models_by_flow[flow]:
        model_index = p.model_index[flow, model_variant]
        static_profile = p.static_profiles_by_variant[model_variant]
        for layer in static_profile.topology.layers_in_topological_order():
            layer_index = p.layer_index[model_variant, layer]
            layer_assignment = assignment.layer_component[
                flow, model_variant, layer, component
            ]
            layer_assignments.append(layer_assignment)
            model.addConstr(
                assignment.component_active[flow, component] >= layer_assignment,
                name=f"active_if_layer[{flow_index},{model_index},{layer_index},{component}]",
            )
    model.addConstr(
        assignment.component_active[flow, component] <= gp.quicksum(layer_assignments),
        name=f"inactive_without_layer[{flow_index},{component}]",
    )


def _component_is_active_if_has_worker(
    model: gp.Model,
    assignment: AssignmentVariables,
    p: CompactFormulationData,
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
    p: CompactFormulationData,
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
    p: CompactFormulationData,
    assignment: AssignmentVariables,
    *,
    flow: FlowId,
    model_variant: ModelVariantId,
    static_profile: ModelVariantStaticProfile,
    flow_index: int,
    model_index: int,
    components: range,
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


def _layer_assignments_respect_precedence(
    model: gp.Model,
    p: CompactFormulationData,
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
                name=f"precedence[{flow_index},{model_index},{source_index},{target_index},{component}]",
            )


def _output_layers_on_last_component(
    model: gp.Model,
    p: CompactFormulationData,
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
                name=f"output_last[{flow_index},{model_index},{layer_index},{component}]",
            )


def _input_layers_on_first_component(
    model: gp.Model,
    p: CompactFormulationData,
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
            name=f"input_first[{flow_index},{model_index},{p.layer_index[model_variant, layer]}]",
        )


def _layer_assigned_to_component_if_model_selected(
    model: gp.Model,
    p: CompactFormulationData,
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
