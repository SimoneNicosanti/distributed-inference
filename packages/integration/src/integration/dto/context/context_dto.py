from uuid import UUID

from pydantic import BaseModel, ConfigDict

from shared.context.model_pass_context import ModelPassType
from shared.flow.flow import FlowId
from shared.plan.partition_replica_id import PartitionReplicaId
from shared.plan.plan_version import PlanVersion
from shared.service.service import ResultSinkId


class ModelInvocationContextDto(BaseModel):
    model_config = ConfigDict(frozen=True)

    flow_id: FlowId
    result_sink: ResultSinkId
    model_invocation_id: UUID


class ModelPassContextDto(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_invocation_context: ModelInvocationContextDto
    plan_version: PlanVersion
    model_pass_type: ModelPassType
    model_pass_id: UUID


class PartitionInvocationContextDto(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_pass_context: ModelPassContextDto
    partition_replica_id: PartitionReplicaId
    partition_invocation_id: UUID


class IngressInvocationContextDto(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_pass_context: ModelPassContextDto
    ingress_invocation_id: UUID


type SourceInvocationContextDto = (
    PartitionInvocationContextDto | IngressInvocationContextDto
)


class PartitionInvocationContributionMetadataDto(BaseModel):
    model_config = ConfigDict(frozen=True)

    source_invocation_context: SourceInvocationContextDto
    target_replica_id: PartitionReplicaId


class PartitionInvocationContributionAckDto(BaseModel):
    model_config = ConfigDict(frozen=True)

    context: SourceInvocationContextDto


class ModelPassResultContributionMetadataDto(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_pass_context: ModelPassContextDto
