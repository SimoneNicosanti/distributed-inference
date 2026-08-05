from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from distributed_inference.domain.identifiers import UserId


class FlowInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    lambda_val: float

    accuracy_req: float
    response_req: float
    energy_req: float


class Flows(BaseModel):
    model_config = ConfigDict(frozen=True)

    flows: list[FlowInfo]


class FlowId(BaseModel):
    model_config = ConfigDict(frozen=True)

    user_id: UserId
    flow_id: UUID = Field(default_factory=uuid4)
