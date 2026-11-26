from collections.abc import Mapping

import gurobipy as gp

from optimizer.adapters.outbound.gurobi.formulation_compiler.formulations.compact.collaborators.assignment.assignment_variables import (
    AssignmentVariables,
)
from optimizer.adapters.outbound.gurobi.formulation_compiler.formulations.compact.collaborators.network.expressions import (
    destination_crossing_expression,
    transmitted_data_size_mb_expression,
)
from optimizer.adapters.outbound.gurobi.formulation_compiler.formulations.compact.collaborators.network.network_variables import (
    NetworkVariables,
)
from optimizer.adapters.outbound.gurobi.gurobi_units import (
    bytes_per_second_to_megabytes_per_second,
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
from shared.model.keys import TensorKey


def add_network_constraints(
    model: gp.Model,
    data: MisocpCompactFormulation,
    *,
    assignment: AssignmentVariables,
    network: NetworkVariables,
    include_shared: bool = True,
) -> None:
    for flow in data.flows:
        for variant in data.configurations_by_flow[flow]:
            _tensor_crossing_cut(
                model,
                data,
                assignment,
                network,
                flow=flow,
                model_variant=variant,
                tensor_routes=data.tensor_routes_by_variant[variant.model_variant_id],
                flow_index=data.flow_index[flow],
                model_index=data.configuration_index[flow, variant],
                cuts=range(1, data.problem.policy.max_components),
            )
    add_transmission_constraints(model, data, assignment=assignment, network=network)
    add_network_queue_constraints(
        model, data, network=network, include_shared=include_shared
    )


def add_transmission_constraints(
    model: gp.Model,
    data: MisocpCompactFormulation,
    *,
    assignment: AssignmentVariables,
    network: NetworkVariables,
) -> None:
    """Bound transmission epigraphs and zero unused sources or empty cuts."""
    problem = data
    max_components = data.problem.policy.max_components
    p = problem
    connections = problem.connections_by_pair
    for flow in p.flows:
        workers = p.problem.flows[flow].workers
        flow_workers = frozenset(workers)
        max_bandwidth_mb_per_s = max(
            (
                bytes_per_second_to_megabytes_per_second(
                    connections[source, target].throughput_bytes_per_sec
                )
                for source in workers
                for target in problem.successors_by_worker[source]
                if target != source and target in flow_workers
            ),
            default=0.0,
        )
        for cut in range(1, max_components):
            _add_latency_constraints(
                model,
                p,
                assignment,
                network,
                flow=flow,
                cut=cut,
            )
            for variant in p.configurations_by_flow[flow]:
                upper = bytes_to_megabytes(
                    problem.problem.configurations[variant].max_cut_size_bytes
                )
                delta = transmitted_data_size_mb_expression(
                    assignment,
                    network,
                    problem,
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
                        name=f"transmission_minimum[{p.flow_index[flow]},{p.configuration_index[flow, variant]},{cut}]",
                    )
                else:
                    model.addConstr(
                        delta == 0,
                        name=f"transmission_no_links[{p.flow_index[flow]},{p.configuration_index[flow, variant]},{cut}]",
                    )
                for source in workers:
                    time = network.transmission_time_s[flow, variant, cut, source]
                    destinations = tuple(
                        target
                        for target in workers
                        if target != source
                        and target in problem.successors_by_worker[source]
                    )
                    if not destinations:
                        time.UB = 0
                        continue
                    inverse_bandwidths = {
                        target: 1
                        / bytes_per_second_to_megabytes_per_second(
                            connections[source, target].throughput_bytes_per_sec
                        )
                        for target in destinations
                    }
                    maximum_inverse_bandwidth = max(inverse_bandwidths.values())
                    time.UB = upper * maximum_inverse_bandwidth
                    model.addConstr(
                        time
                        <= upper
                        * maximum_inverse_bandwidth
                        * assignment.component_worker[flow, cut, source],
                        name=f"compact_transmission_source_upper[{p.flow_index[flow]},{p.configuration_index[flow, variant]},{cut},{p.worker_index[source]}]",
                    )
                    model.addConstr(
                        time <= delta * maximum_inverse_bandwidth,
                        name=f"compact_transmission_cut_upper[{p.flow_index[flow]},{p.configuration_index[flow, variant]},{cut},{p.worker_index[source]}]",
                    )
                    for target in destinations:
                        inverse_bandwidth = inverse_bandwidths[target]
                        model.addConstr(
                            time
                            >= inverse_bandwidth
                            * (
                                delta
                                - upper
                                * (
                                    2
                                    - assignment.component_worker[flow, cut, source]
                                    - assignment.component_worker[flow, cut + 1, target]
                                )
                            ),
                            name=f"compact_transmission[{p.flow_index[flow]},{p.configuration_index[flow, variant]},{cut},{p.worker_index[source]},{p.worker_index[target]}]",
                        )


def add_network_queue_constraints(
    model: gp.Model,
    data: MisocpCompactFormulation,
    *,
    network: NetworkVariables,
    include_shared: bool = True,
    include_local: bool = True,
) -> None:
    problem = data
    max_components = data.problem.policy.max_components
    network_utilization_limit = data.problem.policy.network_utilization_limit
    p = problem
    bins = range(1, len(p.network_bin_limits))
    if include_shared:
        for worker in p.workers:
            selected = tuple(network.network_bin_selected[worker, i] for i in bins)
            rho = network.network_utilization[worker]
            model.addConstr(
                rho
                == gp.quicksum(
                    p.service_arrival_rate_per_s[flow, variant]
                    * network.transmission_time_s[flow, variant, cut, worker]
                    for flow in p.flows
                    if worker in p.problem.flows[flow].workers
                    for variant in p.configurations_by_flow[flow]
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
                    p.network_bin_limits[i - 1]
                    * network.network_bin_selected[worker, i]
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
    if not include_local:
        return
    for flow in p.flows:
        for cut in range(1, max_components):
            responses = []
            for worker in p.problem.flows[flow].workers:
                response = network.source_response_s[flow, cut, worker]
                nominal = gp.quicksum(
                    network.transmission_time_s[flow, variant, cut, worker]
                    for variant in p.configurations_by_flow[flow]
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
    p: MisocpCompactFormulation,
    assignment: AssignmentVariables,
    network: NetworkVariables,
    *,
    flow: FlowId,
    model_variant: ModelConfigurationKey,
    tensor_routes: Mapping[TensorKey, TensorRoute],
    flow_index: int,
    model_index: int,
    cuts: range,
) -> None:
    for tensor, route in tensor_routes.items():
        if len(route.maximal_destinations) <= 1:
            continue
        tensor_index = p.tensor_index[model_variant.model_variant_id, tensor]
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
                destination_index = p.layer_index[
                    model_variant.model_variant_id, destination
                ]
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


def _add_latency_constraints(
    model: gp.Model,
    problem: MisocpCompactFormulation,
    assignment: AssignmentVariables,
    network: NetworkVariables,
    *,
    flow: FlowId,
    cut: int,
) -> None:
    """Exact source latency, independent of model choice and transmitted volume."""
    flow_index = problem.flow_index[flow]
    workers = problem.problem.flows[flow].workers
    for source in workers:
        source_index = problem.worker_index[source]
        placed = assignment.component_worker[flow, cut, source]
        latency = network.source_latency_s[flow, cut, source]
        by_destination = {}
        for target in workers:
            if target == source:
                continue
            target_index = problem.worker_index[target]
            destination = assignment.component_worker[flow, cut + 1, target]
            link = problem.connections_by_pair.get((source, target))
            if link is None:
                model.addConstr(
                    placed + destination <= 1,
                    name=f"missing_transition[{flow_index},{cut},{source_index},{target_index}]",
                )
                continue
            value = link.round_trip_time_s / 2
            by_destination[target] = value
            model.addConstr(
                latency >= value * (placed + destination - 1),
                name=f"latency_lower[{flow_index},{cut},{source_index},{target_index}]",
            )
        upper = max(by_destination.values(), default=0.0)
        model.addConstr(
            latency <= upper * placed,
            name=f"latency_source_upper[{flow_index},{cut},{source_index}]",
        )
        model.addConstr(
            latency
            <= gp.quicksum(
                value * assignment.component_worker[flow, cut + 1, target]
                for target, value in by_destination.items()
            ),
            name=f"latency_destination_upper[{flow_index},{cut},{source_index}]",
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
