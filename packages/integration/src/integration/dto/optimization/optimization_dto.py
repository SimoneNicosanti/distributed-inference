from pydantic import BaseModel, ConfigDict, PositiveFloat

from shared.flow.flow import FlowId
from shared.model.model import ModelTask
from shared.model.model_variant import PerformanceMetric
from shared.service.service import WorkerId


class StreamInfoDto(BaseModel):
    model_config = ConfigDict(frozen=True)

    arrival_rate: PositiveFloat


class QoSRequirementsDto(BaseModel):
    model_config = ConfigDict(frozen=True)

    max_mean_response_time: PositiveFloat
    min_model_performance: PerformanceMetric
    max_energy_consumption: PositiveFloat


class OptimizationRequestDto(BaseModel):
    model_config = ConfigDict(frozen=True)

    flow_id: FlowId
    task: ModelTask
    stream_info: StreamInfoDto
    qos_requirements: QoSRequirementsDto
    ingress_id: WorkerId
