from collections.abc import Mapping
from typing import Annotated

from pydantic import AfterValidator, BaseModel, BeforeValidator, ConfigDict, Field

from integration.dto.optimization.optimization_dto import _require_explicit_worker_id
from integration.shared.id_json import service_id_from_json, service_id_to_json
from shared.flow.flow import FlowId
from shared.model.model_variant import PerformanceMetricType
from shared.service.service import WorkerId


class OptimizerDto(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


type PositiveInt = Annotated[int, Field(strict=True, gt=0)]
type NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
type FiniteFloat = Annotated[float, Field(strict=True, allow_inf_nan=False)]
type NonNegativeFloat = Annotated[float, Field(strict=True, ge=0, allow_inf_nan=False)]
type PositiveFloat = Annotated[float, Field(strict=True, gt=0, allow_inf_nan=False)]


def _canonical_worker(value: WorkerId) -> WorkerId:
    return service_id_from_json(service_id_to_json(value))


def _require_explicit_flow_id(value: object) -> object:
    if isinstance(value, Mapping):
        user = value.get("user_id")
        if "flow_id" not in value or "user_id" not in value:
            raise ValueError("flow_id must contain explicit flow_id and user_id fields")
        if isinstance(user, Mapping) and "id" not in user:
            raise ValueError("user_id must contain an explicit id")
    return value


type WorkerReference = Annotated[
    WorkerId,
    BeforeValidator(_require_explicit_worker_id),
    AfterValidator(_canonical_worker),
]
type FlowReference = Annotated[FlowId, BeforeValidator(_require_explicit_flow_id)]


class PerformanceMetricDto(OptimizerDto):
    type: PerformanceMetricType
    dataset: str
    value: FiniteFloat
