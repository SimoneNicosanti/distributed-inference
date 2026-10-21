from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, model_validator

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


class OptimizeInferenceDistributionRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    requests: Annotated[tuple[OptimizationRequestDto, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def validate_unique_flow_ids(self) -> Self:
        flow_ids = tuple(request.flow_id for request in self.requests)
        if len(flow_ids) != len(set(flow_ids)):
            raise ValueError("Optimization requests must have unique flow IDs")
        return self
