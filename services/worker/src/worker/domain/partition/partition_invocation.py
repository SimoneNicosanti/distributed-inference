from pydantic import BaseModel, ConfigDict

from shared.context.model_pass_context import ModelPassContext
from shared.flow.flow import FlowId
from shared.model.model_partition import ModelPartitionId
from shared.plan.partition_replica_id import PartitionReplicaId
from shared.plan.plan_version import PlanVersion
from shared.tensor.tensor_bundle import TensorBundle
from worker.domain.context.partition_invocation_context import (
    PartitionInvocationContext,
)


## This is the input of the partition.
## It contains only the inputs of the partition, not the skip tensors.
class PartitionInvocationRequest(BaseModel):
    model_config = ConfigDict(frozen=True)
    context: PartitionInvocationContext

    payload: TensorBundle

    @property
    def plan_version(self) -> PlanVersion:
        return self.context.model_pass_context.plan_version

    @property
    def partition_id(self) -> ModelPartitionId:
        return self.context.partition_replica_id.partition_id

    @property
    def partition_replica_id(self) -> PartitionReplicaId:
        return self.context.partition_replica_id

    @property
    def flow_id(self) -> FlowId:
        return self.context.model_pass_context.model_invocation_context.flow_id


## This is the output of the sub-model as returned to the caller
class PartitionInvocationResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    context: PartitionInvocationContext

    payload: TensorBundle

    @property
    def model_pass_context(self) -> ModelPassContext:
        return self.context.model_pass_context
