from pydantic import BaseModel, ConfigDict

from distributed_inference.domain.model_graph_info import TaskType


class FlowInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    lambda_val: float

    accuracy_req: float
    response_req: float
    energy_req: float

    task: TaskType


class Flows(BaseModel):
    model_config = ConfigDict(frozen=True)

    flows: list[FlowInfo]
