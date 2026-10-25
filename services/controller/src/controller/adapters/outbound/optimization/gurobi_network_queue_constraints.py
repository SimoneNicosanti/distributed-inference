import gurobipy as gp
from gurobipy import GRB

from controller.adapters.outbound.optimization.gurobi_problem import (
    GurobiProblem,
)
from controller.adapters.outbound.optimization.gurobi_variables import (
    NetworkVariables,
)


def add_network_queue_constraints(
    model: gp.Model,
    problem: GurobiProblem,
    network: NetworkVariables,
    *,
    max_components: int,
    network_utilization_limit: float,
) -> None:
    p = problem
    bin_count = len(p.network_bin_limits) - 1
    cuts = range(1, max_components)

    for worker in p.workers:
        worker_index = p.worker_index[worker]
        relevant_flows = tuple(
            flow for flow in p.flows if worker in p.workers_by_flow[flow]
        )
        model.addConstr(
            network.network_utilization[worker]
            == gp.quicksum(
                p.service_arrival_rate_per_s[flow, model_variant]
                * network.transmission_time_s[flow, model_variant, cut, worker]
                for flow in relevant_flows
                for model_variant in p.models_by_flow[flow]
                for cut in cuts
            ),
            name=f"network_utilization_value[{worker_index}]",
        )
        model.addConstr(
            network.network_utilization[worker] <= network_utilization_limit,
            name=f"network_stability[{worker_index}]",
        )
        model.addConstr(
            gp.quicksum(
                network.network_bin_selected[worker, bin_index]
                for bin_index in range(1, bin_count + 1)
            )
            == 1,
            name=f"choose_network_bin[{worker_index}]",
        )
        model.addConstr(
            gp.quicksum(
                p.network_bin_limits[bin_index - 1]
                * network.network_bin_selected[worker, bin_index]
                for bin_index in range(1, bin_count + 1)
            )
            <= network.network_utilization[worker],
            name=f"network_bin_lower[{worker_index}]",
        )
        model.addConstr(
            network.network_utilization[worker]
            <= gp.quicksum(
                p.network_bin_limits[bin_index]
                * network.network_bin_selected[worker, bin_index]
                for bin_index in range(1, bin_count + 1)
            ),
            name=f"network_bin_upper[{worker_index}]",
        )

    for flow in p.flows:
        flow_index = p.flow_index[flow]
        for cut in cuts:
            response = network.component_transmission_response_s[flow, cut]
            for worker in p.workers_by_flow[flow]:
                worker_index = p.worker_index[worker]
                nominal = gp.quicksum(
                    network.transmission_time_s[flow, model_variant, cut, worker]
                    for model_variant in p.models_by_flow[flow]
                )
                for bin_index in range(1, bin_count + 1):
                    model.addGenConstrIndicator(
                        network.network_bin_selected[worker, bin_index],
                        True,
                        (1 - p.network_bin_limits[bin_index]) * response - nominal,
                        GRB.GREATER_EQUAL,
                        0.0,
                        name=(
                            f"network_response[{flow_index},{cut},{worker_index},"
                            f"{bin_index}]"
                        ),
                    )
