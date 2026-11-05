import math

import gurobipy as gp
from gurobipy import GRB

from controller.adapters.outbound.optimization.gurobi_problem import (
    GurobiProblem,
)
from controller.adapters.outbound.optimization.gurobi_variables import (
    AssignmentVariables,
    QoSVariables,
)
from controller.domain.optimization.request.optimization_request import (
    OptimizationRequestsByFlow,
)
from controller.domain.optimization.solving.optimization_context import (
    OptimizationContext,
)
from controller.domain.optimization.solving.solver_config import ObjectiveKind
from shared.model.model_variant import PerformanceMetricType


def set_objective(
    model: gp.Model,
    problem: GurobiProblem,
    qos: QoSVariables,
    objective_kind: ObjectiveKind,
    *,
    requests_by_flow: OptimizationRequestsByFlow,
    assignment: AssignmentVariables,
    context: OptimizationContext,
) -> None:
    p = problem
    arrival_rates = {
        flow: float(requests_by_flow[flow].stream_info.arrival_rate) for flow in p.flows
    }
    total_arrival_rate = sum(arrival_rates.values())

    sense = GRB.MINIMIZE
    if objective_kind == ObjectiveKind.ARRIVAL_WEIGHTED_RESPONSE_TIME:
        objective = gp.quicksum(
            arrival_rates[flow] / total_arrival_rate * qos.response_time_s[flow]
            for flow in p.flows
        )
    elif objective_kind == ObjectiveKind.ARRIVAL_WEIGHTED_ACCURACY:
        coefficients = {}
        for flow in p.flows:
            requirement = requests_by_flow[flow].qos_requirements.min_model_performance
            if requirement.type not in {
                PerformanceMetricType.ACCURACY,
                PerformanceMetricType.MAP_AT_50,
                PerformanceMetricType.MAP_AT_75,
            }:
                raise ValueError(
                    "Accuracy objective requires an accuracy or mAP requirement; "
                    f"got {requirement.type.value!r}"
                )
            for variant in p.models_by_flow[flow]:
                values = [
                    float(metric.value)
                    for metric in context.static_profiles_by_variant[
                        variant
                    ].info.accuracies
                    if metric.type == requirement.type
                    and metric.dataset == requirement.dataset
                    and math.isfinite(metric.value)
                ]
                if not values:
                    raise ValueError(f"Missing matching accuracy for {variant!r}")
                coefficients[flow, variant] = max(values)
        objective = gp.quicksum(
            arrival_rates[flow]
            / total_arrival_rate
            * coefficients[flow, variant]
            * assignment.model_selected[flow, variant]
            for flow in p.flows
            for variant in p.models_by_flow[flow]
        )
        sense = GRB.MAXIMIZE
    elif objective_kind == ObjectiveKind.DEADLINE_VIOLATIONS:
        violations = []
        for flow in p.flows:
            index = p.flow_index[flow]
            violated = model.addVar(
                lb=0.0, ub=1.0, vtype=GRB.BINARY, name=f"deadline_violation[{index}]"
            )
            model.addGenConstrIndicator(
                violated,
                False,
                qos.response_time_s[flow],
                GRB.LESS_EQUAL,
                float(requests_by_flow[flow].qos_requirements.max_mean_response_time),
                name=f"deadline_if_not_violated[{index}]",
            )
            violations.append(violated)
        objective = gp.quicksum(violations)
    elif objective_kind == ObjectiveKind.ARRIVAL_WEIGHTED_NORMALIZED_DELAY:
        delays = {}
        for flow in p.flows:
            index = p.flow_index[flow]
            delay = model.addVar(lb=0.0, name=f"response_delay[{index}]")
            model.addConstr(
                delay
                >= qos.response_time_s[flow]
                - float(requests_by_flow[flow].qos_requirements.max_mean_response_time),
                name=f"response_delay_lower[{index}]",
            )
            delays[flow] = delay
        objective = gp.quicksum(
            arrival_rates[flow]
            / total_arrival_rate
            * (
                delays[flow]
                / float(requests_by_flow[flow].qos_requirements.max_mean_response_time)
            )
            for flow in p.flows
        )
    else:
        raise AssertionError(f"Unknown objective: {objective_kind!r}")

    model.setObjective(objective, sense)
