from pydantic import BaseModel, ConfigDict, NonNegativeInt

from shared.flow.flow import FlowId
from shared.model.keys import TensorKey
from shared.model.model_partition import ModelPartitionId
from shared.model.model_variant import ModelVariantId
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


class ReplicaInputBindingDto(BaseModel):
    model_config = ConfigDict(frozen=True)

    prev_replica_id: PartitionReplicaId | None
    tensors: tuple[TensorKey, ...]


class ReplicaOutputRouteDto(BaseModel):
    model_config = ConfigDict(frozen=True)

    next_replica_id: PartitionReplicaId | None
    tensors: tuple[TensorKey, ...]


class ReplicaRoutingDto(BaseModel):
    model_config = ConfigDict(frozen=True)

    input_bindings: tuple[ReplicaInputBindingDto, ...]
    output_routes: tuple[ReplicaOutputRouteDto, ...]


class RoutingPlanDto(BaseModel):
    model_config = ConfigDict(frozen=True)

    replica_ids: tuple[PartitionReplicaId, ...]
    routings: tuple[ReplicaRoutingDto, ...]


class PriorityKeyDto(BaseModel):
    model_config = ConfigDict(frozen=True)

    flow_id: FlowId
    partition_replica_id: PartitionReplicaId


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
    routing_plan: RoutingPlanDto


class IngressInputRouteDto(BaseModel):
    model_config = ConfigDict(frozen=True)

    target_replica_id: PartitionReplicaId
    tensors: tuple[TensorKey, ...]


class FlowPlanDto(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_variant_id: ModelVariantId
    input_routes: tuple[IngressInputRouteDto, ...]
    result_tensors: tuple[TensorKey, ...]


class IngressPlanDto(BaseModel):
    model_config = ConfigDict(frozen=True)

    plan_version: PlanVersion
    flow_ids: tuple[FlowId, ...]
    flow_plans: tuple[FlowPlanDto, ...]


class ServicePlanDto(BaseModel):
    model_config = ConfigDict(frozen=True)

    worker_plan: WorkerPlanDto
    ingress_plan: IngressPlanDto | None
