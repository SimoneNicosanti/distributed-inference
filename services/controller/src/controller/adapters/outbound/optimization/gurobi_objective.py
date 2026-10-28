import gurobipy as gp
from gurobipy import GRB

from controller.adapters.outbound.optimization.gurobi_problem import (
    GurobiProblem,
)
from controller.adapters.outbound.optimization.gurobi_variables import (
    QoSVariables,
)
from controller.domain.optimization.optimization_request import (
    OptimizationRequestsByFlow,
)
from controller.domain.optimization.solver_config import ObjectiveKind


def set_objective(
    model: gp.Model,
    problem: GurobiProblem,
    qos: QoSVariables,
    objective_kind: ObjectiveKind,
    *,
    requests_by_flow: OptimizationRequestsByFlow,
) -> None:
    p = problem
    arrival_rates = {
        flow: float(requests_by_flow[flow].stream_info.arrival_rate) for flow in p.flows
    }
    total_arrival_rate = sum(arrival_rates.values())

    if objective_kind == ObjectiveKind.ARRIVAL_WEIGHTED_RESPONSE_SLACK:
        objective = gp.quicksum(
            arrival_rates[flow]
            / total_arrival_rate
            * (
                qos.response_time_s[flow]
                # - float(requests_by_flow[flow].qos_requirements.max_mean_response_time)
            )
            for flow in p.flows
        )
    elif objective_kind == ObjectiveKind.ARRIVAL_WEIGHTED_NORMALIZED_RESPONSE_SLACK:
        objective = gp.quicksum(
            arrival_rates[flow]
            / total_arrival_rate
            * (
                qos.response_time_s[flow]
                / float(requests_by_flow[flow].qos_requirements.max_mean_response_time)
                - 1
            )
            for flow in p.flows
        )
    else:
        raise AssertionError(f"Unknown objective: {objective_kind!r}")

    model.setObjective(objective, GRB.MINIMIZE)
