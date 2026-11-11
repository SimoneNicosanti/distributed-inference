from dataclasses import dataclass

import gurobipy as gp

from shared.flow.flow import FlowId
from shared.model.model_variant import ModelVariantId
from shared.service.service import WorkerId


@dataclass(slots=True)
class ExecutionVariables:
    # Its square enters the compute-queue SOC, so it remains explicit.
    execution_time_s: dict[tuple[FlowId, ModelVariantId, int, WorkerId], gp.Var]
