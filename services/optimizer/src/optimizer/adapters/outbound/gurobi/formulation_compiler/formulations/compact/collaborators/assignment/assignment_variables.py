from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from typing import cast, override

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

type LayerComponentKey = tuple[FlowId, ModelConfigurationKey, LayerKey, int]
type AssignmentTerm = gp.Var | gp.LinExpr


@dataclass(slots=True)
class AssignmentVariables:
    max_components: int
    configuration_selected: dict[FlowConfiguration, gp.Var]
    layer_component: Mapping[LayerComponentKey, AssignmentTerm]
    component_worker: dict[tuple[FlowId, int, WorkerId], gp.Var]
    component_active: dict[tuple[FlowId, int], gp.Var]
    component_last: dict[tuple[FlowId, int], gp.Var]

    def extend(self, added: AssignmentVariables) -> None:
        """Append flow assignments while retaining existing native variables."""
        if self.max_components != added.max_components:
            raise ValueError("Cannot change max_components while extending assignments")
        self.configuration_selected.update(added.configuration_selected)
        if isinstance(self.layer_component, MutableMapping):
            self.layer_component.update(added.layer_component)
        else:
            self.layer_component = {**self.layer_component, **added.layer_component}
        self.component_worker.update(added.component_worker)
        self.component_active.update(added.component_active)
        self.component_last.update(added.component_last)

    def layer_prefix_at(
        self,
        flow: FlowId,
        configuration: ModelConfigurationKey,
        layer: LayerKey,
        cut: int,
    ) -> AssignmentTerm:
        return gp.quicksum(
            self.layer_component[flow, configuration, layer, component]
            for component in range(1, cut + 1)
        )

    def layer_decision_variables(self) -> tuple[gp.Var, ...]:
        return tuple(
            cast(gp.Var, variable) for variable in self.layer_component.values()
        )

    def layer_decision_items(self) -> tuple[tuple[LayerComponentKey, gp.Var], ...]:
        return tuple(
            (key, cast(gp.Var, variable))
            for key, variable in self.layer_component.items()
        )

    def layer_start_variables(
        self,
        flow: FlowId,
        configuration: ModelConfigurationKey,
        layer: LayerKey,
        component: int,
    ) -> tuple[gp.Var, ...]:
        return (
            cast(gp.Var, self.layer_component[flow, configuration, layer, component]),
        )

    def component_active_at(self, flow: FlowId, component: int) -> gp.Var | gp.LinExpr:
        """Return the activation variable, or zero for the terminal sentinel."""
        if component == self.max_components + 1:
            return gp.LinExpr(0.0)
        if component < 1 or component > self.max_components:
            raise IndexError(f"component index out of range: {component}")
        return self.component_active[flow, component]


@dataclass(slots=True)
class CumulativeAssignmentVariables(AssignmentVariables):
    layer_cumulative: dict[LayerComponentKey, gp.Var]

    @override
    def extend(self, added: AssignmentVariables) -> None:
        if not isinstance(added, CumulativeAssignmentVariables):
            raise ValueError("Cannot mix assignment representations")
        AssignmentVariables.extend(self, added)
        self.layer_cumulative.update(added.layer_cumulative)

    @override
    def layer_prefix_at(
        self,
        flow: FlowId,
        configuration: ModelConfigurationKey,
        layer: LayerKey,
        cut: int,
    ) -> AssignmentTerm:
        if cut == 0:
            return gp.LinExpr(0.0)
        if cut == self.max_components:
            return self.configuration_selected[flow, configuration]
        return self.layer_cumulative[flow, configuration, layer, cut]

    @override
    def layer_decision_variables(self) -> tuple[gp.Var, ...]:
        return tuple(self.layer_cumulative.values())

    @override
    def layer_decision_items(self) -> tuple[tuple[LayerComponentKey, gp.Var], ...]:
        return tuple(self.layer_cumulative.items())

    @override
    def layer_start_variables(
        self,
        flow: FlowId,
        configuration: ModelConfigurationKey,
        layer: LayerKey,
        component: int,
    ) -> tuple[gp.Var, ...]:
        return tuple(
            self.layer_cumulative[flow, configuration, layer, cut]
            for cut in range(component, self.max_components)
        )
