from dataclasses import dataclass

import gurobipy as gp

from optimizer.domain.formulation.compact.misocp.misocp_compact_formulation import (
    FlowConfiguration,
)
from optimizer.domain.problem.compact.compact_optimization_problem import (
    ModelConfigurationKey,
)
from shared.flow.flow import FlowId
from shared.model.keys import LayerKey
from shared.service.service import WorkerId


@dataclass(slots=True)
class AssignmentVariables:
    max_components: int
    configuration_selected: dict[FlowConfiguration, gp.Var]
    layer_component: dict[tuple[FlowId, ModelConfigurationKey, LayerKey, int], gp.Var]
    component_worker: dict[tuple[FlowId, int, WorkerId], gp.Var]
    component_active: dict[tuple[FlowId, int], gp.Var]
    component_last: dict[tuple[FlowId, int], gp.Var]

    def component_active_at(self, flow: FlowId, component: int) -> gp.Var | gp.LinExpr:
        """Return the activation variable, or zero for the terminal sentinel."""
        if component == self.max_components + 1:
            return gp.LinExpr(0.0)
        if component < 1 or component > self.max_components:
            raise IndexError(f"component index out of range: {component}")
        return self.component_active[flow, component]
