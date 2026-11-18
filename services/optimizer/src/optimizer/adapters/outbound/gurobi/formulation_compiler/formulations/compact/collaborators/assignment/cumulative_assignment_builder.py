from typing import override

import gurobipy as gp
from gurobipy import GRB

from optimizer.adapters.outbound.gurobi.formulation_compiler.formulations.compact.collaborators.assignment.assignment_builder import (
    CompactAssignmentBuilder,
)
from optimizer.adapters.outbound.gurobi.formulation_compiler.formulations.compact.collaborators.assignment.assignment_variables import (
    AssignmentTerm,
    CumulativeAssignmentVariables,
    LayerComponentKey,
)
from optimizer.domain.formulation.compact.misocp.misocp_compact_formulation import (
    FlowConfiguration,
    MisocpCompactFormulation,
)
from shared.flow.flow import FlowId
from shared.service.service import WorkerId


class CumulativeAssignmentBuilder(CompactAssignmentBuilder):
    """Build y directly; layer_component contains only derived linear expressions."""

    @override
    def create_variables(
        self, model: gp.Model, data: MisocpCompactFormulation
    ) -> CumulativeAssignmentVariables:
        count = data.problem.policy.max_components
        selected: dict[FlowConfiguration, gp.Var] = {}
        cumulative: dict[LayerComponentKey, gp.Var] = {}
        assignments: dict[LayerComponentKey, AssignmentTerm] = {}
        workers: dict[tuple[FlowId, int, WorkerId], gp.Var] = {}
        active: dict[tuple[FlowId, int], gp.Var] = {}
        last: dict[tuple[FlowId, int], gp.Var] = {}
        for flow in data.flows:
            f = data.flow_index[flow]
            for configuration in data.configurations_by_flow[flow]:
                m = data.configuration_index[flow, configuration]
                # With K=1 there is no integer layer variable to imply u's integrality.
                u = selected[flow, configuration] = model.addVar(
                    vtype=GRB.BINARY, name=f"u[{f},{m}]"
                )
                for layer in data.problem.variants[
                    configuration.model_variant_id
                ].layers:
                    layer_index = data.layer_index[
                        configuration.model_variant_id, layer
                    ]
                    previous: AssignmentTerm = gp.LinExpr(0.0)
                    for component in range(1, count + 1):
                        current: AssignmentTerm
                        if component == count:
                            current = u
                        else:
                            current = model.addVar(
                                vtype=GRB.BINARY,
                                name=f"y[{f},{m},{layer_index},{component}]",
                            )
                            cumulative[flow, configuration, layer, component] = current
                        assignments[flow, configuration, layer, component] = (
                            current - previous
                        )
                        previous = current
            for component in range(1, count + 1):
                active[flow, component] = model.addVar(
                    lb=0.0, ub=1.0, name=f"active[{f},{component}]"
                )
                last[flow, component] = model.addVar(
                    lb=0.0, ub=1.0, name=f"last[{f},{component}]"
                )
                for worker in data.problem.flows[flow].workers:
                    s = data.worker_index[worker]
                    workers[flow, component, worker] = model.addVar(
                        lb=0.0, ub=1.0, name=f"e[{f},{component},{s}]"
                    )
        return CumulativeAssignmentVariables(
            count, selected, assignments, workers, active, last, cumulative
        )
