from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, NonNegativeFloat

from shared.user.user import UserId


class FlowInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    lambda_val: NonNegativeFloat

    accuracy_req: NonNegativeFloat
    response_req: NonNegativeFloat
    energy_req: NonNegativeFloat
    throughput_req: NonNegativeFloat


class FlowId(BaseModel):
    model_config = ConfigDict(frozen=True)

    user_id: UserId
    flow_id: UUID = Field(default_factory=uuid4)


class Flow(BaseModel):
    model_config = ConfigDict(frozen=True)

    flow_id: FlowId
    flow_info: FlowInfo
