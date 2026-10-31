from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from shared.flow.flow import FlowId
from shared.plan.hop_logical_id import (
    IngressReplicaId,
    PartitionReplicaId,
    SinkReplicaId,
)
from shared.plan.plan_version import PlanVersion


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
    ingress_replica_id: IngressReplicaId


class PartitionReplicaHopDto(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["partition_replica"] = "partition_replica"
    partition_replica_id: PartitionReplicaId


class SinkHopDto(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["sink"] = "sink"
    sink_replica_id: SinkReplicaId


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
