from typing import Self

from pydantic import BaseModel, ConfigDict, NonNegativeInt, model_validator

from integration.dto.context.context_dto import (
    DestinationHopDto,
    InvocableHopDto,
    SourceHopDto,
)
from shared.flow.flow import FlowId
from shared.model.keys import TensorKey
from shared.model.model_partition import ModelPartitionId
from shared.plan.hop_logical_id import (
    PlannedIngressReplicaId,
    PlannedPartitionReplicaId,
    PlannedSinkReplicaId,
)
from shared.plan.plan_version import PlanVersion


class ResourceAllocationDto(BaseModel):
    model_config = ConfigDict(frozen=True)

    use_gpu: bool


class DeploymentPlanDto(BaseModel):
    model_config = ConfigDict(frozen=True)

    executable_hops: tuple[PlannedPartitionReplicaId, ...]
    allocations: tuple[ResourceAllocationDto, ...]
    ingress_hops: tuple[PlannedIngressReplicaId, ...]
    sink_hops: tuple[PlannedSinkReplicaId, ...]


class PartitionIOSchemeDto(BaseModel):
    model_config = ConfigDict(frozen=True)

    inputs: tuple[TensorKey, ...]
    outputs: tuple[TensorKey, ...]


class PartitionExecutionPlanDto(BaseModel):
    model_config = ConfigDict(frozen=True)

    partition_ids: tuple[ModelPartitionId, ...]
    schemes: tuple[PartitionIOSchemeDto, ...]


class SourceHopAddressDto(BaseModel):
    model_config = ConfigDict(frozen=True)

    flow_id: FlowId
    hop: SourceHopDto


class DestinationHopAddressDto(BaseModel):
    model_config = ConfigDict(frozen=True)

    flow_id: FlowId
    hop: DestinationHopDto


class InvocableHopAddressDto(BaseModel):
    model_config = ConfigDict(frozen=True)

    flow_id: FlowId
    hop: InvocableHopDto


class HopInputRoutingDto(BaseModel):
    model_config = ConfigDict(frozen=True)

    source: SourceHopAddressDto
    tensors: tuple[TensorKey, ...]


class HopOutputRoutingDto(BaseModel):
    model_config = ConfigDict(frozen=True)

    destination: DestinationHopAddressDto
    tensors: tuple[TensorKey, ...]


class HopRoutingsDto(BaseModel):
    model_config = ConfigDict(frozen=True)

    input_routings: tuple[HopInputRoutingDto, ...]
    output_routings: tuple[HopOutputRoutingDto, ...]


class HopRoutingPlanDto(BaseModel):
    model_config = ConfigDict(frozen=True)

    hop_addresses: tuple[InvocableHopAddressDto, ...]
    routings: tuple[HopRoutingsDto, ...]


class PriorityKeyDto(BaseModel):
    model_config = ConfigDict(frozen=True)

    flow_id: FlowId
    executable_hop: PlannedPartitionReplicaId


class SchedulingPlanDto(BaseModel):
    model_config = ConfigDict(frozen=True)

    priority_keys: tuple[PriorityKeyDto, ...]
    priorities: tuple[NonNegativeInt, ...]


class FailurePlanDto(BaseModel):
    model_config = ConfigDict(frozen=True)

    flow_ids: tuple[FlowId, ...]
    sink_replica_ids: tuple[PlannedSinkReplicaId, ...]

    @model_validator(mode="after")
    def validate_flow_sinks(self) -> Self:
        if len(self.flow_ids) != len(self.sink_replica_ids):
            raise ValueError("Each flow must have a failure sink")
        if len(self.flow_ids) != len(set(self.flow_ids)):
            raise ValueError("Failure plan flows must be unique")
        return self


class WorkerPlanDto(BaseModel):
    model_config = ConfigDict(frozen=True)

    plan_version: PlanVersion
    deployment_plan: DeploymentPlanDto
    scheduling_plan: SchedulingPlanDto
    partition_execution_plan: PartitionExecutionPlanDto
    hop_routing_plan: HopRoutingPlanDto
    failure_plan: FailurePlanDto
