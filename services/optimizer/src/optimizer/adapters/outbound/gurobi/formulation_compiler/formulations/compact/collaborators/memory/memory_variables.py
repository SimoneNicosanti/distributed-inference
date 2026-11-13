from dataclasses import dataclass

import gurobipy as gp

from optimizer.domain.problem.compact.compact_optimization_problem import (
    ModelConfigurationKey,
)
from shared.flow.flow import FlowId
from shared.service.service import WorkerId


@dataclass(slots=True)
class MemoryVariables:
    component_weight_mb: dict[
        tuple[FlowId, ModelConfigurationKey, int, WorkerId], gp.Var
    ]
    max_activation_mb: dict[WorkerId, gp.Var]
