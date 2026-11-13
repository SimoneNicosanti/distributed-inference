from dataclasses import dataclass

import gurobipy as gp

from optimizer.domain.problem.compact.compact_optimization_problem import (
    ModelConfigurationKey,
)
from shared.flow.flow import FlowId
from shared.service.service import WorkerId


@dataclass(slots=True)
class ExecutionVariables:
    # Its square enters the compute-queue SOC, so it remains explicit.
    execution_time_s: dict[tuple[FlowId, ModelConfigurationKey, int, WorkerId], gp.Var]
