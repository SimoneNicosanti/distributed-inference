import gurobipy as gp

from controller.adapters.outbound.optimization.gurobi.model_builder.formulations.compact.collaborators.assignment.assignment_variables import (
    AssignmentVariables,
)
from controller.adapters.outbound.optimization.gurobi.model_builder.formulations.compact.collaborators.network.expressions import (
    destination_crossing_expression,
    transmitted_data_size_mb_expression,
)
from controller.adapters.outbound.optimization.gurobi.model_builder.formulations.compact.collaborators.network.network_variables import (
    NetworkVariables,
)
from controller.adapters.outbound.optimization.gurobi.model_builder.formulations.compact.profile_bounds import (
    cut_upper_mb,
)
from controller.adapters.outbound.optimization.gurobi.gurobi_units import (
    bytes_per_second_to_megabytes_per_second,
)
from controller.domain.optimization.solving.formulation.compact.compact_formulation_data import (
    CompactFormulationData,
)
from controller.domain.profile.model_variant_static_profile import (
    ModelVariantStaticProfile,
)
from controller.domain.profile.network_topology import (
    NetworkTopology,
)
from shared.flow.flow import FlowId
from shared.model.model_variant import ModelVariantId


def add_network_constraints(
    model: gp.Model,
    data: CompactFormulationData,
    *,
    assignment: AssignmentVariables,
    network: NetworkVariables,
) -> None:
    for flow in data.flows:
        for variant in data.models_by_flow[flow]:
            _tensor_crossing_cut(
                model,
                data,
                assignment,
                network,
                flow=flow,
                model_variant=variant,
                static_profile=data.static_profiles_by_variant[variant],
                flow_index=data.flow_index[flow],
                model_index=data.model_index[flow, variant],
                cuts=range(1, data.max_components),
            )
    add_transmission_constraints(model, data, assignment=assignment, network=network)
    add_network_queue_constraints(model, data, network=network)


def add_transmission_constraints(
    model: gp.Model,
    data: CompactFormulationData,
    *,
    assignment: AssignmentVariables,
    network: NetworkVariables,
) -> None:
    problem = data
    max_components = data.max_components
    p = problem
    connections = problem.network_topology.connections()
    for flow in p.flows:
        workers = p.workers_by_flow[flow]
        flow_workers = frozenset(workers)
        max_bandwidth_mb_per_s = max(
            (
                bytes_per_second_to_megabytes_per_second(
                    connections[source, target].throughput_bytes_per_sec
                )
                for source in workers
                for target in problem.network_topology.successors(source)
                if target != source and target in flow_workers
            ),
            default=0.0,
        )
        for cut in range(1, max_components):
            _add_transition_marginal_constraints(
                model,
                p,
                assignment,
                network,
                topology=problem.network_topology,
                flow=flow,
                cut=cut,
            )
            for variant in p.models_by_flow[flow]:
                profile = problem.static_profiles_by_variant[variant]
                upper = cut_upper_mb(profile)
                delta = transmitted_data_size_mb_expression(
                    assignment,
                    network,
                    profile,
                    flow=flow,
                    model_variant=variant,
                    cut=cut,
                )
                if max_bandwidth_mb_per_s > 0:
                    model.addConstr(
                        gp.quicksum(
                            network.transmission_time_s[flow, variant, cut, source]
                            for source in workers
                        )
                        >= delta / max_bandwidth_mb_per_s,
                        name=f"transmission_minimum[{p.flow_index[flow]},{p.model_index[flow, variant]},{cut}]",
                    )
                else:
                    model.addConstr(
                        delta == 0,
                        name=f"transmission_no_links[{p.flow_index[flow]},{p.model_index[flow, variant]},{cut}]",
                    )
                for source in workers:
                    time = network.transmission_time_s[flow, variant, cut, source]
                    destinations = tuple(
                        target
                        for target in workers
                        if target != source
                        and target in problem.network_topology.successors(source)
                    )
                    for target in destinations:
                        bandwidth = bytes_per_second_to_megabytes_per_second(
                            connections[source, target].throughput_bytes_per_sec
                        )
                        model.addConstr(
                            time
                            >= (
                                delta
                                - upper
                                * (
                                    2
                                    - assignment.component_worker[flow, cut, source]
                                    - assignment.component_worker[flow, cut + 1, target]
                                )
                            )
                            / bandwidth,
                            name=f"compact_transmission[{p.flow_index[flow]},{p.model_index[flow, variant]},{cut},{p.worker_index[source]},{p.worker_index[target]}]",
                        )
                    if not destinations:
                        time.UB = 0


def add_network_queue_constraints(
    model: gp.Model, data: CompactFormulationData, *, network: NetworkVariables
) -> None:
    problem = data
    max_components = data.max_components
    network_utilization_limit = data.network_utilization_limit
    p = problem
    bins = range(1, len(p.network_bin_limits))
    for worker in p.workers:
        selected = tuple(network.network_bin_selected[worker, i] for i in bins)
        rho = network.network_utilization[worker]
        model.addConstr(
            rho
            == gp.quicksum(
                p.service_arrival_rate_per_s[flow, variant]
                * network.transmission_time_s[flow, variant, cut, worker]
                for flow in p.flows
                if worker in p.workers_by_flow[flow]
                for variant in p.models_by_flow[flow]
                for cut in range(1, max_components)
            ),
            name=f"network_utilization_value[{p.worker_index[worker]}]",
        )
        model.addConstr(
            rho <= network_utilization_limit,
            name=f"network_stability[{p.worker_index[worker]}]",
        )
        model.addConstr(
            gp.quicksum(selected) == 1,
            name=f"choose_network_bin[{p.worker_index[worker]}]",
        )
        model.addConstr(
            gp.quicksum(
                p.network_bin_limits[i - 1] * network.network_bin_selected[worker, i]
                for i in bins
            )
            <= rho,
            name=f"network_bin_lower[{p.worker_index[worker]}]",
        )
        model.addConstr(
            rho
            <= gp.quicksum(
                p.network_bin_limits[i] * network.network_bin_selected[worker, i]
                for i in bins
            ),
            name=f"network_bin_upper[{p.worker_index[worker]}]",
        )
    for flow in p.flows:
        for cut in range(1, max_components):
            responses = []
            for worker in p.workers_by_flow[flow]:
                response = network.source_response_s[flow, cut, worker]
                nominal = gp.quicksum(
                    network.transmission_time_s[flow, variant, cut, worker]
                    for variant in p.models_by_flow[flow]
                )
                add_projected_ps_epigraph(
                    model,
                    response,
                    nominal,
                    network.nominal_upper_s[flow, worker],
                    p.network_bin_limits,
                    tuple(network.network_bin_selected[worker, i] for i in bins),
                    name=f"compact_ps[{p.flow_index[flow]},{cut},{p.worker_index[worker]}]",
                )
                responses.append(response)
            model.addConstr(
                network.component_transmission_response_s[flow, cut]
                >= gp.quicksum(responses),
                name=f"compact_ps_sum[{p.flow_index[flow]},{cut}]",
            )


def _tensor_crossing_cut(
    model: gp.Model,
    p: CompactFormulationData,
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


def _add_transition_marginal_constraints(
    model: gp.Model,
    problem: CompactFormulationData,
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


def add_projected_ps_epigraph(
    model: gp.Model,
    response: gp.Var,
    nominal: gp.LinExpr | gp.Var,
    upper: float,
    limits: tuple[float, ...],
    selected: tuple[gp.Var, ...],
    *,
    name: str,
) -> None:
    multipliers = tuple(1 / (1 - limit) for limit in limits[1:])
    model.addConstr(nominal <= upper, name=f"{name}_bound")
    for i, multiplier in enumerate(multipliers):
        correction = gp.quicksum(
            (multiplier - multipliers[j]) * selected[j] for j in range(i)
        )
        model.addConstr(
            response >= multiplier * nominal - upper * correction,
            name=f"{name}[{i + 1}]",
        )
