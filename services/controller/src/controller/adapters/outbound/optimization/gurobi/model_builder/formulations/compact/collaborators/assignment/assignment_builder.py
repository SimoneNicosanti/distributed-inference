import gurobipy as gp
from gurobipy import GRB

from controller.adapters.outbound.optimization.gurobi.model_builder.formulations.compact.collaborators.assignment.assignment_variables import (
    AssignmentVariables,
)
from controller.domain.optimization.solving.formulation.compact.compact_formulation_data import (
    CompactFormulationData,
    FlowModel,
)
from shared.flow.flow import FlowId
from shared.model.keys import LayerKey
from shared.model.model_variant import ModelVariantId
from shared.service.service import WorkerId

from .constraints import add_assignment_constraints


class CompactAssignmentBuilder:
    def create_variables(
        self, model: gp.Model, data: CompactFormulationData
    ) -> AssignmentVariables:
        variables = _add_assignment_variables(
            model, data, max_components=data.max_components
        )
        for placement in variables.component_worker.values():
            placement.VType = GRB.CONTINUOUS
            placement.LB = 0
            placement.UB = 1
        return variables

    def add_constraints(
        self,
        model: gp.Model,
        data: CompactFormulationData,
        *,
        assignment: AssignmentVariables,
    ) -> None:
        add_assignment_constraints(model, data, assignment=assignment)


def _add_assignment_variables(
    model: gp.Model,
    problem: CompactFormulationData,
    *,
    max_components: int,
) -> AssignmentVariables:
    model_selected: dict[FlowModel, gp.Var] = {}
    layer_component: dict[tuple[FlowId, ModelVariantId, LayerKey, int], gp.Var] = {}
    component_worker: dict[tuple[FlowId, int, WorkerId], gp.Var] = {}
    component_active: dict[tuple[FlowId, int], gp.Var] = {}
    component_last: dict[tuple[FlowId, int], gp.Var] = {}
    components = range(1, max_components + 1)
    for flow in problem.flows:
        flow_index = problem.flow_index[flow]
        for model_variant in problem.models_by_flow[flow]:
            model_index = problem.model_index[flow, model_variant]
            static_profile = problem.static_profiles_by_variant[model_variant]
            model_selected[flow, model_variant] = model.addVar(
                lb=0.0,
                ub=1.0,
                name=f"u[{flow_index},{model_index}]",
            )
            for layer in static_profile.topology.layers_in_topological_order():
                layer_index = problem.layer_index[model_variant, layer]
                for component in components:
                    layer_component[flow, model_variant, layer, component] = (
                        model.addVar(
                            vtype=GRB.BINARY,
                            name=(
                                f"x[{flow_index},{model_index},{layer_index},"
                                f"{component}]"
                            ),
                        )
                    )
        for component in components:
            component_active[flow, component] = model.addVar(
                lb=0.0,
                ub=1.0,
                name=f"active[{flow_index},{component}]",
            )
            component_last[flow, component] = model.addVar(
                lb=0.0,
                ub=1.0,
                name=f"last[{flow_index},{component}]",
            )
            for worker in problem.workers_by_flow[flow]:
                component_worker[flow, component, worker] = model.addVar(
                    vtype=GRB.BINARY,
                    name=(
                        f"e[{flow_index},{component},{problem.worker_index[worker]}]"
                    ),
                )

    return AssignmentVariables(
        max_components=max_components,
        model_selected=model_selected,
        layer_component=layer_component,
        component_worker=component_worker,
        component_active=component_active,
        component_last=component_last,
    )
