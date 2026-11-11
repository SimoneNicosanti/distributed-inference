from dataclasses import dataclass

import gurobipy as gp

from shared.flow.flow import FlowId


@dataclass(slots=True)
class QoSVariables:
    response_time_s: dict[FlowId, gp.Var]
