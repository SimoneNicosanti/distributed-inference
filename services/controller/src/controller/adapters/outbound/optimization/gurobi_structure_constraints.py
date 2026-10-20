import gurobipy as gp
from gurobipy import GRB

from controller.adapters.outbound.optimization.gurobi_problem import (
    GurobiProblem,
)
from controller.adapters.outbound.optimization.gurobi_profile_data import (
    layer_execution_profiles,
    tensor_routes,
    tensor_size_bytes,
)
from controller.adapters.outbound.optimization.gurobi_variables import (
    AssignmentVariables,
    ExecutionVariables,
    NetworkVariables,
)
from controller.domain.optimization.optimization_context import OptimizationContext
from controller.domain.optimization.optimization_request import (
    OptimizationRequestsByFlow,
)
from controller.domain.profile.model_variant_static_profile import (
    ModelVariantStaticProfile,
)
from shared.flow.flow import FlowId
from shared.model.model_variant import ModelVariantId


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
        model.addConstr(
            gp.quicksum(
                assignment.model_selected[flow, model_variant]
                for model_variant in p.models_by_flow[flow]
            )
            == 1,
            name=f"choose_model[{flow_index}]",
        )
        for component in components:
            model.addConstr(
                assignment.component_active[flow, component]
                == gp.quicksum(
                    assignment.component_worker[flow, component, worker]
                    for worker in p.workers_by_flow[flow]
                ),
                name=f"active_worker[{flow_index},{component}]",
            )
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
                        assignment.component_active[flow, component]
                        >= layer_assignment,
                        name=(
                            f"active_if_layer[{flow_index},{model_index},"
                            f"{layer_index},{component}]"
                        ),
                    )
            model.addConstr(
                assignment.component_active[flow, component]
                <= gp.quicksum(layer_assignments),
                name=f"inactive_without_layer[{flow_index},{component}]",
            )
            if component < max_components:
                model.addConstr(
                    assignment.component_last[flow, component]
                    == assignment.component_active[flow, component]
                    - assignment.component_active[flow, component + 1],
                    name=f"last_component[{flow_index},{component}]",
                )
                model.addConstr(
                    assignment.component_active[flow, component + 1]
                    <= assignment.component_active[flow, component],
                    name=f"active_prefix[{flow_index},{component}]",
                )
            else:
                model.addConstr(
                    assignment.component_last[flow, component]
                    == assignment.component_active[flow, component],
                    name=f"last_component[{flow_index},{component}]",
                )

        ingress = requests_by_flow[flow].ingress_id
        model.addConstr(
            assignment.component_worker[flow, 1, ingress] == 1,
            name=f"first_on_device[{flow_index}]",
        )
        for component in components:
            model.addConstr(
                assignment.component_worker[flow, component, ingress]
                >= assignment.component_last[flow, component],
                name=f"last_on_device[{flow_index},{component}]",
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
                max_components=max_components,
                components=components,
                cuts=cuts,
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
    max_components: int,
    components: range,
    cuts: range,
) -> None:
    layer_descriptions = static_profile.topology.layers()
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
    for tensor, route in tensor_routes(static_profile).items():
        tensor_index = p.tensor_index[model_variant, tensor]
        source = route.source
        for cut in cuts:
            crosses = network.tensor_crosses_cut[flow, model_variant, tensor, cut]
            source_before = gp.quicksum(
                assignment.layer_component[flow, model_variant, source, component]
                for component in range(1, cut + 1)
            )
            destinations_after = gp.quicksum(
                assignment.layer_component[flow, model_variant, destination, component]
                for destination in route.destinations
                for component in range(cut + 1, max_components + 1)
            )
            model.addConstr(
                crosses <= source_before,
                name=f"cut_source[{flow_index},{model_index},{tensor_index},{cut}]",
            )
            model.addConstr(
                crosses <= destinations_after,
                name=(
                    f"cut_destination[{flow_index},{model_index},{tensor_index},{cut}]"
                ),
            )
            for destination in route.destinations:
                destination_index = p.layer_index[model_variant, destination]
                destination_after = gp.quicksum(
                    assignment.layer_component[
                        flow, model_variant, destination, component
                    ]
                    for component in range(cut + 1, max_components + 1)
                )
                model.addConstr(
                    crosses >= source_before + destination_after - 1,
                    name=(
                        f"cut_exact[{flow_index},{model_index},{tensor_index},"
                        f"{cut},{destination_index}]"
                    ),
                )


def add_execution_constraints(
    model: gp.Model,
    problem: GurobiProblem,
    assignment: AssignmentVariables,
    execution: ExecutionVariables,
    *,
    context: OptimizationContext,
    max_components: int,
) -> None:
    p = problem
    for flow in p.flows:
        flow_index = p.flow_index[flow]
        for model_variant in p.models_by_flow[flow]:
            model_index = p.model_index[flow, model_variant]
            static_profile = context.static_profiles_by_variant[model_variant]
            layers = static_profile.topology.layers_in_topological_order()
            profiles_by_worker = {
                worker: layer_execution_profiles(context, model_variant, worker)
                for worker in p.workers_by_flow[flow]
            }
            for component in range(1, max_components + 1):
                for worker in p.workers_by_flow[flow]:
                    worker_index = p.worker_index[worker]
                    execution_profiles = profiles_by_worker[worker]
                    execution_terms: list[gp.LinExpr] = []
                    for layer in layers:
                        layer_index = p.layer_index[model_variant, layer]
                        z = assignment.layer_component_worker[
                            flow, model_variant, layer, component, worker
                        ]
                        x = assignment.layer_component[
                            flow, model_variant, layer, component
                        ]
                        e = assignment.component_worker[flow, component, worker]
                        model.addConstr(
                            z <= x,
                            name=(
                                f"z_x[{flow_index},{model_index},{layer_index},"
                                f"{component},{worker_index}]"
                            ),
                        )
                        model.addConstr(
                            z <= e,
                            name=(
                                f"z_e[{flow_index},{model_index},{layer_index},"
                                f"{component},{worker_index}]"
                            ),
                        )
                        model.addConstr(
                            z >= x + e - 1,
                            name=(
                                f"z_exact[{flow_index},{model_index},{layer_index},"
                                f"{component},{worker_index}]"
                            ),
                        )
                        layer_profile = (
                            execution_profiles.get(layer)
                            if execution_profiles is not None
                            else None
                        )
                        if layer_profile is None:
                            model.addConstr(
                                z == 0,
                                name=(
                                    f"unsupported[{flow_index},{model_index},"
                                    f"{layer_index},{component},{worker_index}]"
                                ),
                            )
                        else:
                            execution_terms.append(
                                float(layer_profile.execution_time_s) * z
                            )
                    model.addConstr(
                        execution.execution_time[flow, model_variant, component, worker]
                        == gp.quicksum(execution_terms),
                        name=(
                            f"execution_time[{flow_index},{model_index},{component},"
                            f"{worker_index}]"
                        ),
                    )


def add_transmission_constraints(
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
    connections = context.network_topology.connections()
    for flow in p.flows:
        flow_index = p.flow_index[flow]
        workers = p.workers_by_flow[flow]
        response_limit = float(
            requests_by_flow[flow].qos_requirements.max_mean_response_time
        )
        for cut in range(1, max_components):
            for target in workers:
                allowed_sources = tuple(
                    source
                    for source in workers
                    if (flow, cut, source, target) in network.adjacent_component_workers
                )
                if len(allowed_sources) < len(workers):
                    model.addConstr(
                        assignment.component_worker[flow, cut + 1, target]
                        <= gp.quicksum(
                            assignment.component_worker[flow, cut, source]
                            for source in allowed_sources
                        ),
                        name=(
                            f"available_predecessor[{flow_index},{cut},"
                            f"{p.worker_index[target]}]"
                        ),
                    )

            for source in workers:
                source_index = p.worker_index[source]
                for target in workers:
                    target_index = p.worker_index[target]
                    current = assignment.component_worker[flow, cut, source]
                    following = assignment.component_worker[flow, cut + 1, target]
                    q = network.adjacent_component_workers.get(
                        (flow, cut, source, target)
                    )
                    if q is None:
                        continue
                    model.addConstr(
                        q <= current,
                        name=(
                            f"q_source[{flow_index},{cut},{source_index},"
                            f"{target_index}]"
                        ),
                    )
                    model.addConstr(
                        q <= following,
                        name=(
                            f"q_target[{flow_index},{cut},{source_index},"
                            f"{target_index}]"
                        ),
                    )
                    model.addConstr(
                        q >= current + following - 1,
                        name=(
                            f"q_exact[{flow_index},{cut},{source_index},{target_index}]"
                        ),
                    )

            for model_variant in p.models_by_flow[flow]:
                model_index = p.model_index[flow, model_variant]
                static_profile = context.static_profiles_by_variant[model_variant]
                routes = tensor_routes(static_profile)
                delta = gp.quicksum(
                    tensor_size_bytes(static_profile, tensor)
                    * network.tensor_crosses_cut[flow, model_variant, tensor, cut]
                    for tensor in routes
                )
                for source in workers:
                    source_index = p.worker_index[source]
                    transmission = network.transmission_time[
                        flow, model_variant, cut, source
                    ]
                    outgoing = gp.quicksum(
                        network.adjacent_component_workers[flow, cut, source, target]
                        for target in workers
                        if (flow, cut, source, target)
                        in network.adjacent_component_workers
                    )
                    # The draft states ``1 - sum(q)`` here, which would force
                    # an active transmission to zero.  ``sum(q)`` instead
                    # deactivates only sources that do not host the component.
                    model.addConstr(
                        transmission <= response_limit * outgoing,
                        name=(
                            f"transmission_active[{flow_index},{model_index},{cut},"
                            f"{source_index}]"
                        ),
                    )
                    for target in workers:
                        q = network.adjacent_component_workers.get(
                            (flow, cut, source, target)
                        )
                        if q is None:
                            continue
                        target_index = p.worker_index[target]
                        if source == target:
                            model.addGenConstrIndicator(
                                q,
                                True,
                                transmission,
                                GRB.EQUAL,
                                0.0,
                                name=(
                                    f"local_transmission[{flow_index},{model_index},"
                                    f"{cut},{source_index}]"
                                ),
                            )
                        else:
                            throughput = connections[
                                source, target
                            ].throughput_bytes_per_sec
                            model.addGenConstrIndicator(
                                q,
                                True,
                                throughput * transmission - delta,
                                GRB.EQUAL,
                                0.0,
                                name=(
                                    f"transmission_value[{flow_index},{model_index},"
                                    f"{cut},{source_index},{target_index}]"
                                ),
                            )
