from typing import cast

import gurobipy as gp
import pytest

from controller.adapters.outbound.optimization.gurobi_variables import (
    AssignmentVariables,
)
from shared.flow.flow import FlowId


def _assignment_variables(
    flow: FlowId, active: gp.Var, *, max_components: int = 2
) -> AssignmentVariables:
    return AssignmentVariables(
        max_components=max_components,
        model_selected={},
        layer_component={},
        component_worker={},
        component_active={(flow, 1): active},
        component_last={},
    )


def test_component_active_accessor_returns_variable_and_terminal_sentinel() -> None:
    flow = cast(FlowId, object())
    active = cast(gp.Var, object())
    variables = _assignment_variables(flow, active)

    assert variables.component_active_at(flow, 1) is active
    sentinel = variables.component_active_at(flow, 3)
    assert isinstance(sentinel, gp.LinExpr)
    assert sentinel.size() == 0
    assert sentinel.getConstant() == 0.0


@pytest.mark.parametrize("component", [0, 4])
def test_component_active_accessor_rejects_indices_outside_model(
    component: int,
) -> None:
    flow = cast(FlowId, object())
    variables = _assignment_variables(flow, cast(gp.Var, object()))

    with pytest.raises(IndexError, match="component index out of range"):
        variables.component_active_at(flow, component)
