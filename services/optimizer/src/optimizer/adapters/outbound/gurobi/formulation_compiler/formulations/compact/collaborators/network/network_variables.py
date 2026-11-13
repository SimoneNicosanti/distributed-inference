from dataclasses import dataclass

import gurobipy as gp

from optimizer.domain.problem.compact.compact_optimization_problem import (
    ModelConfigurationKey,
)
from shared.flow.flow import FlowId
from shared.model.keys import TensorKey
from shared.service.service import WorkerId


@dataclass(slots=True)
class NetworkVariables:
    tensor_crossing_auxiliary: dict[
        tuple[FlowId, ModelConfigurationKey, TensorKey, int], gp.Var
    ]
    adjacent_component_workers: dict[tuple[FlowId, int, WorkerId, WorkerId], gp.Var]
    transmission_time_s: dict[
        tuple[FlowId, ModelConfigurationKey, int, WorkerId], gp.Var
    ]
    network_utilization: dict[WorkerId, gp.Var]
    network_bin_selected: dict[tuple[WorkerId, int], gp.Var]
    component_transmission_response_s: dict[tuple[FlowId, int], gp.Var]
    source_response_s: dict[tuple[FlowId, int, WorkerId], gp.Var]
    nominal_upper_s: dict[tuple[FlowId, WorkerId], float]
