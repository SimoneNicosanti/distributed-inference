from pydantic import BaseModel, ConfigDict

from controller.domain.optimization.qos_constraints import QoSRequirements
from controller.domain.optimization.stream_info import StreamInfo
from shared.flow.flow import FlowId
from shared.model.model import ModelTask


class OptimizationRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    flow_id: FlowId
    task: ModelTask

    stream_info: StreamInfo
    qos_requirements: QoSRequirements
