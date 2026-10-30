from pydantic import BaseModel, ConfigDict, NonNegativeInt

from integration.dto.context.context_dto import (
    DestinationHopDto,
    InvocableHopDto,
    PartitionReplicaHopDto,
    SourceHopDto,
)
from shared.flow.flow import FlowId
from shared.model.keys import TensorKey
from shared.model.model_partition import ModelPartitionId
from shared.plan.partition_replica_id import PartitionReplicaId
from shared.plan.plan_version import PlanVersion


class ResourceAllocationDto(BaseModel):
    model_config = ConfigDict(frozen=True)

    use_gpu: bool


class DeploymentPlanDto(BaseModel):
    model_config = ConfigDict(frozen=True)

    replica_ids: tuple[PartitionReplicaId, ...]
    allocations: tuple[ResourceAllocationDto, ...]


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
    executable_hop: PartitionReplicaHopDto


class SchedulingPlanDto(BaseModel):
    model_config = ConfigDict(frozen=True)

    priority_keys: tuple[PriorityKeyDto, ...]
    priorities: tuple[NonNegativeInt, ...]


class WorkerPlanDto(BaseModel):
    model_config = ConfigDict(frozen=True)

    plan_version: PlanVersion
    deployment_plan: DeploymentPlanDto
    scheduling_plan: SchedulingPlanDto
    partition_execution_plan: PartitionExecutionPlanDto
    hop_routing_plan: HopRoutingPlanDto
