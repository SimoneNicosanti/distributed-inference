import math

import gurobipy as gp
from gurobipy import GRB

from controller.adapters.outbound.optimization.gurobi.model_builder.formulations.compact.collaborators.assignment.assignment_variables import (
    AssignmentVariables,
)
from controller.adapters.outbound.optimization.gurobi.model_builder.formulations.compact.collaborators.response_time.response_time_variables import (
    QoSVariables,
)
from controller.domain.optimization.solving.formulation.compact.compact_formulation_data import (
    CompactFormulationData,
)
from controller.domain.optimization.solving.formulation.compact.compact_objective_kind import (
    CompactObjectiveKind,
)
from shared.model.model_variant import PerformanceMetricType


class CompactObjectiveBuilder:
    def set_objective(
        self,
        model: gp.Model,
        data: CompactFormulationData,
        *,
        assignment: AssignmentVariables,
        qos: QoSVariables,
    ) -> None:
        problem = data
        objective_kind = data.objective_kind
        p = problem
        arrival_rates = {
            flow: float(problem.arrival_rate_per_s[flow]) for flow in p.flows
        }
        total_arrival_rate = sum(arrival_rates.values())
        sense = GRB.MINIMIZE
        objective: gp.LinExpr | int
        if objective_kind == CompactObjectiveKind.ARRIVAL_WEIGHTED_RESPONSE_TIME:
            objective = gp.quicksum(
                arrival_rates[flow] / total_arrival_rate * qos.response_time_s[flow]
                for flow in p.flows
            )
        elif objective_kind == CompactObjectiveKind.ARRIVAL_WEIGHTED_ACCURACY:
            coefficients = {}
            for flow in p.flows:
                requirement = problem.performance_requirement_by_flow[flow]
                if requirement.type not in {
                    PerformanceMetricType.ACCURACY,
                    PerformanceMetricType.MAP_AT_50,
                    PerformanceMetricType.MAP_AT_75,
                }:
                    raise ValueError(
                        f"Accuracy objective requires an accuracy or mAP requirement; got {requirement.type.value!r}"
                    )
                for variant in p.models_by_flow[flow]:
                    values = [
                        float(metric.value)
                        for metric in problem.static_profiles_by_variant[
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
        elif objective_kind == CompactObjectiveKind.DEADLINE_VIOLATIONS:
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
                    float(problem.max_response_time_s_by_flow[flow]),
                    name=f"deadline_if_not_violated[{index}]",
                )
                violations.append(violated)
            objective = gp.quicksum(violations)
        elif objective_kind == CompactObjectiveKind.ARRIVAL_WEIGHTED_NORMALIZED_DELAY:
            delays = {}
            for flow in p.flows:
                index = p.flow_index[flow]
                delay = model.addVar(lb=0.0, name=f"response_delay[{index}]")
                model.addConstr(
                    delay
                    >= qos.response_time_s[flow]
                    - float(problem.max_response_time_s_by_flow[flow]),
                    name=f"response_delay_lower[{index}]",
                )
                delays[flow] = delay
            objective = gp.quicksum(
                arrival_rates[flow]
                / total_arrival_rate
                * (delays[flow] / float(problem.max_response_time_s_by_flow[flow]))
                for flow in p.flows
            )
        elif objective_kind == CompactObjectiveKind.CONSTANT:
            objective = 0
        else:
            raise AssertionError(f"Unknown objective: {objective_kind!r}")
        model.setObjective(objective, sense)
