from contextlib import suppress
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field

from shared.flow.flow import FlowId
from shared.plan.partition_replica_id import PartitionReplicaId
from shared.plan.plan_version import PlanVersion
from shared.service.service import ServiceId


def _canonical_service_id(value: object) -> object:
    if isinstance(value, ServiceId):
        raw_id = value.id
    elif isinstance(value, dict) and "id" in value:
        raw_id = value["id"]
    else:
        return value

    if isinstance(raw_id, str):
        with suppress(ValueError):
            raw_id = UUID(raw_id)

    return ServiceId(id=raw_id)


type HopServiceIdDto = Annotated[
    ServiceId,
    BeforeValidator(_canonical_service_id),
]


class ModelInvocationContextDto(BaseModel):
    model_config = ConfigDict(frozen=True)

    flow_id: FlowId
    model_invocation_id: UUID


class ModelPassContextDto(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_invocation_context: ModelInvocationContextDto
    plan_version: PlanVersion
    model_pass_type: Literal["forward"]
    model_pass_id: UUID


class ExternalHopDto(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["external"] = "external"


class IngressHopDto(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["ingress"] = "ingress"
    ingress_id: HopServiceIdDto


class PartitionReplicaHopDto(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["partition_replica"] = "partition_replica"
    partition_replica_id: PartitionReplicaId


class SinkHopDto(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["sink"] = "sink"
    sink_id: HopServiceIdDto


type SourceHopDto = Annotated[
    ExternalHopDto | IngressHopDto | PartitionReplicaHopDto,
    Field(discriminator="kind"),
]
type DestinationHopDto = Annotated[
    IngressHopDto | PartitionReplicaHopDto | SinkHopDto,
    Field(discriminator="kind"),
]
type InvocableHopDto = DestinationHopDto
type HopDto = Annotated[
    ExternalHopDto | IngressHopDto | PartitionReplicaHopDto | SinkHopDto,
    Field(discriminator="kind"),
]


class HopInvocationContextDto(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_pass_context: ModelPassContextDto
    hop: SourceHopDto
    hop_invocation_id: UUID


class HopInvocationContributionMetadataDto(BaseModel):
    model_config = ConfigDict(frozen=True)

    source_context: HopInvocationContextDto
    destination_hop: DestinationHopDto


class HopInvocationContributionAckDto(BaseModel):
    model_config = ConfigDict(frozen=True)

    source_context: HopInvocationContextDto
    destination_hop: DestinationHopDto


class ModelPassResultMetadataDto(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_pass_context: ModelPassContextDto
