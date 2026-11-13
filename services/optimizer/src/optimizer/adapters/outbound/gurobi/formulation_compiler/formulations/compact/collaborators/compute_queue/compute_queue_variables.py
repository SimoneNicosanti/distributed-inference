from dataclasses import dataclass

import gurobipy as gp

from shared.flow.flow import FlowId
from shared.service.service import WorkerId


@dataclass(slots=True)
class ComputeQueueVariables:
    component_priority: dict[tuple[FlowId, int, WorkerId, int], gp.Var]
    priority_used: dict[tuple[WorkerId, int], gp.Var]
    priority_class_utilization: dict[tuple[FlowId, int, WorkerId, int], gp.Var]
    higher_priority_utilization: dict[tuple[WorkerId, int], gp.Var]
    compute_bin_selected: dict[tuple[WorkerId, int, int], gp.Var]
    queue_wait_bin_s: dict[tuple[WorkerId, int, int], gp.Var]
    queue_wait_s: dict[tuple[WorkerId, int], gp.Var]
    weighted_queue_wait_s: dict[tuple[WorkerId, int], gp.Var]
    remaining_compute_capacity: dict[tuple[WorkerId, int], gp.Var]
    component_queue_wait_s: dict[tuple[FlowId, int], gp.Var]
