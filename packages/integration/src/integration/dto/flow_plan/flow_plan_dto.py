from pydantic import BaseModel, ConfigDict

from shared.model.model_variant import ModelVariantId
from shared.plan.hop_logical_id import IngressReplicaId, SinkReplicaId
from shared.plan.plan_version import PlanVersion


class InputSpecDto(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    shape: tuple[int | str | None, ...]
    dtype: str


class ModelSpecsDto(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_variant_id: ModelVariantId
    inputs: tuple[InputSpecDto, ...]


class FlowPlanDto(BaseModel):
    model_config = ConfigDict(frozen=True)

    plan_version: PlanVersion
    model_specs: ModelSpecsDto
    ingress_replica_id: IngressReplicaId
    sink_replica_id: SinkReplicaId
