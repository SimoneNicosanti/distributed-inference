"""Strong projected PS epigraph, with no bin-disaggregation variables."""

from dataclasses import dataclass

import gurobipy as gp

from controller.adapters.outbound.optimization.compact_transmission import cut_upper_mb
from controller.adapters.outbound.optimization.gurobi_problem import GurobiProblem
from controller.adapters.outbound.optimization.gurobi_units import (
    bytes_per_second_to_megabytes_per_second,
)
from controller.adapters.outbound.optimization.gurobi_variables import NetworkVariables
from controller.domain.optimization.solving.optimization_context import (
    OptimizationContext,
)
from shared.flow.flow import FlowId
from shared.service.service import WorkerId


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


def add_compact_network_queue_constraints(
    model: gp.Model,
    problem: GurobiProblem,
    network: NetworkVariables,
    *,
    max_components: int,
    network_utilization_limit: float,
) -> None:
    # Bounds are attached during variable creation from the same profile/link
    # data; the shared NetworkVariables API remains sufficient for extraction.
    if not isinstance(network, CompactNetworkVariables):
        raise TypeError("compact PS queues require CompactNetworkVariables")
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


@dataclass(slots=True)
class CompactNetworkVariables(NetworkVariables):
    source_response_s: dict[tuple[FlowId, int, WorkerId], gp.Var]
    nominal_upper_s: dict[tuple[FlowId, WorkerId], float]


def add_compact_network_variables(
    model: gp.Model,
    problem: GurobiProblem,
    base: NetworkVariables,
    context: OptimizationContext,
    *,
    max_components: int,
) -> CompactNetworkVariables:
    upper = {}
    responses = {}
    connections = context.network_topology.connections()
    for flow in problem.flows:
        for worker in problem.workers_by_flow[flow]:
            destinations = tuple(
                target
                for target in problem.workers_by_flow[flow]
                if target != worker
                and target in context.network_topology.successors(worker)
            )
            upper[flow, worker] = max(
                (
                    cut_upper_mb(context.static_profiles_by_variant[variant])
                    / bytes_per_second_to_megabytes_per_second(
                        connections[worker, target].throughput_bytes_per_sec
                    )
                    for variant in problem.models_by_flow[flow]
                    for target in destinations
                ),
                default=0.0,
            )
            for cut in range(1, max_components):
                responses[flow, cut, worker] = model.addVar(
                    lb=0,
                    name=f"rtx_source[{problem.flow_index[flow]},{cut},{problem.worker_index[worker]}]",
                )
    return CompactNetworkVariables(
        tensor_crossing_auxiliary=base.tensor_crossing_auxiliary,
        adjacent_component_workers=base.adjacent_component_workers,
        transmission_time_s=base.transmission_time_s,
        network_utilization=base.network_utilization,
        network_bin_selected=base.network_bin_selected,
        component_transmission_response_s=base.component_transmission_response_s,
        source_response_s=responses,
        nominal_upper_s=upper,
    )
