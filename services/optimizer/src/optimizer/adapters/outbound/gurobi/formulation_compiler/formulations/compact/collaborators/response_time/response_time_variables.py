from dataclasses import dataclass, field

import gurobipy as gp

from shared.flow.flow import FlowId


@dataclass(slots=True)
class QoSVariables:
    response_time_s: dict[FlowId, gp.Var]
    response_delay_s: dict[FlowId, gp.Var] = field(default_factory=dict)
    deadline_violation: dict[FlowId, gp.Var] = field(default_factory=dict)
