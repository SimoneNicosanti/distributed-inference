from dataclasses import dataclass

import gurobipy as gp

from shared.flow.flow import FlowId
from shared.model.model_variant import ModelVariantId
from shared.service.service import WorkerId


@dataclass(slots=True)
class MemoryVariables:
    component_weight_mb: dict[tuple[FlowId, ModelVariantId, int, WorkerId], gp.Var]
    max_activation_mb: dict[WorkerId, gp.Var]
