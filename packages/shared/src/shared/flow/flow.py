import math
from dataclasses import dataclass, field
from uuid import UUID, uuid4

from shared.user.user import UserId


@dataclass(frozen=True, slots=True)
class FlowInfo:
    name: str
    lambda_val: float

    accuracy_req: float
    response_req: float
    energy_req: float
    throughput_req: float

    def __post_init__(self) -> None:
        values = (
            self.lambda_val,
            self.accuracy_req,
            self.response_req,
            self.energy_req,
            self.throughput_req,
        )
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("Flow requirements must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class FlowId:
    user_id: UserId
    flow_id: UUID = field(default_factory=uuid4)


@dataclass(frozen=True, slots=True)
class Flow:
    flow_id: FlowId
    flow_info: FlowInfo
