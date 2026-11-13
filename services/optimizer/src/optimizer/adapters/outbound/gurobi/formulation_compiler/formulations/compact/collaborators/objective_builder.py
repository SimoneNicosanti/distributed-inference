import math

import gurobipy as gp
from gurobipy import GRB

from optimizer.adapters.outbound.gurobi.formulation_compiler.formulations.compact.collaborators.assignment.assignment_variables import (
    AssignmentVariables,
)
from optimizer.adapters.outbound.gurobi.formulation_compiler.formulations.compact.collaborators.response_time.response_time_variables import (
    QoSVariables,
)
from optimizer.domain.formulation.compact.misocp.misocp_compact_formulation import (
    MisocpCompactFormulation,
)
from optimizer.domain.problem.compact.compact_optimization_problem import (
    ObjectiveKind,
)
from shared.model.model_variant import PerformanceMetricType


class CompactObjectiveBuilder:
    def set_objective(
        self,
        model: gp.Model,
        data: MisocpCompactFormulation,
        *,
        assignment: AssignmentVariables,
        qos: QoSVariables,
    ) -> None:
        problem = data
        objective_kind = data.problem.policy.objective
        p = problem
        arrival_rates = {
            flow: float(problem.problem.flows[flow].arrival_rate_per_s)
            for flow in p.flows
        }
        total_arrival_rate = sum(arrival_rates.values())
        sense = GRB.MINIMIZE
        objective: gp.LinExpr | int
        if objective_kind == ObjectiveKind.ARRIVAL_WEIGHTED_RESPONSE_TIME:
            objective = gp.quicksum(
                arrival_rates[flow] / total_arrival_rate * qos.response_time_s[flow]
                for flow in p.flows
            )
        elif objective_kind == ObjectiveKind.ARRIVAL_WEIGHTED_ACCURACY:
            coefficients = {}
            for flow in p.flows:
                requirement = problem.problem.flows[flow].performance_requirement
                if requirement.type not in {
                    PerformanceMetricType.ACCURACY,
                    PerformanceMetricType.MAP_AT_50,
                    PerformanceMetricType.MAP_AT_75,
                }:
                    raise ValueError(
                        f"Accuracy objective requires an accuracy or mAP requirement; got {requirement.type.value!r}"
                    )
                for variant in p.configurations_by_flow[flow]:
                    values = [
                        float(metric.value)
                        for metric in problem.problem.variants[
                            variant.model_variant_id
                        ].accuracies
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
                * assignment.configuration_selected[flow, variant]
                for flow in p.flows
                for variant in p.configurations_by_flow[flow]
            )
            sense = GRB.MAXIMIZE
        elif objective_kind == ObjectiveKind.DEADLINE_VIOLATIONS:
            violations = []
            for flow in p.flows:
                index = p.flow_index[flow]
                violated = model.addVar(
                    lb=0.0,
                    ub=1.0,
                    vtype=GRB.BINARY,
                    name=f"deadline_violation[{index}]",
                )
                model.addGenConstrIndicator(
                    violated,
                    False,
                    qos.response_time_s[flow],
                    GRB.LESS_EQUAL,
                    float(problem.problem.flows[flow].max_response_time_s),
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
                    - float(problem.problem.flows[flow].max_response_time_s),
                    name=f"response_delay_lower[{index}]",
                )
                delays[flow] = delay
            objective = gp.quicksum(
                arrival_rates[flow]
                / total_arrival_rate
                * (
                    delays[flow]
                    / float(problem.problem.flows[flow].max_response_time_s)
                )
                for flow in p.flows
            )
        elif objective_kind == ObjectiveKind.CONSTANT:
            objective = 0
        else:
            raise AssertionError(f"Unknown objective: {objective_kind!r}")
        model.setObjective(objective, sense)
