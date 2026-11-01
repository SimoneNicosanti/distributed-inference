from collections.abc import Mapping
from typing import Annotated, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, PositiveFloat

from shared.flow.flow import FlowId
from shared.model.model import ModelTask
from shared.model.model_variant import PerformanceMetric
from shared.service.service import WorkerId


def _require_explicit_worker_id(value: object) -> object:
    if isinstance(value, Mapping) and "id" not in value:
        raise ValueError("worker_id must contain an explicit id")
    return value


class StreamInfoDto(BaseModel):
    model_config = ConfigDict(frozen=True)

    arrival_rate: PositiveFloat


class QoSRequirementsDto(BaseModel):
    model_config = ConfigDict(frozen=True)

    max_mean_response_time: PositiveFloat
    min_model_performance: PerformanceMetric
    max_energy_consumption: PositiveFloat


class FixedPlacementDto(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["fixed"] = "fixed"
    worker_id: Annotated[WorkerId, BeforeValidator(_require_explicit_worker_id)]


class AnyPlacementDto(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["any"] = "any"


type PlacementDto = Annotated[
    FixedPlacementDto | AnyPlacementDto,
    Field(discriminator="kind"),
]


class FlowEndpointPlacementDto(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    ingress_placement: PlacementDto
    sink_placement: PlacementDto


class ValidLocalWorkerConfigDto(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["valid"] = "valid"
    worker_id: Annotated[WorkerId, BeforeValidator(_require_explicit_worker_id)]
    is_inference_runner: bool


class NoLocalWorkerConfigDto(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["none"] = "none"


type LocalWorkerConfigDto = Annotated[
    ValidLocalWorkerConfigDto | NoLocalWorkerConfigDto,
    Field(discriminator="kind"),
]


class OptimizationRequestDto(BaseModel):
    model_config = ConfigDict(frozen=True)

    flow_id: FlowId
    task: ModelTask
    stream_info: StreamInfoDto
    qos_requirements: QoSRequirementsDto
    flow_endpoint_placement: FlowEndpointPlacementDto
    local_worker_config: LocalWorkerConfigDto
