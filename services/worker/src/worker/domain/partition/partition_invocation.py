from pydantic import BaseModel, ConfigDict

from shared.flow.flow import FlowId
from shared.model.model_partition import ModelPartitionId
from shared.plan.plan import InferencePlanVersion, PartitionDeployment
from worker.domain.context.partition_invocation_context import (
    PartitionInvocationContext,
)
from worker.domain.partition.tensor_bundle import TensorBundle


## This is the input of the sub-model once everything for the sub-model has been gathered from predecessors
class PartitionInvocationRequest(BaseModel):
    model_config = ConfigDict(frozen=True)
    context: PartitionInvocationContext

    payload: TensorBundle

    @property
    def plan_version(self) -> InferencePlanVersion:
        return self.context.model_pass_context.plan_version

    @property
    def partition_id(self) -> ModelPartitionId:
        return self.context.partition_deployment_id.partition_id

    @property
    def flow_id(self) -> FlowId:
        return self.context.model_pass_context.model_invocation_context.flow_id

    @property
    def partition_deployment(self) -> PartitionDeployment:
        return self.context.partition_deployment_id


## This is the output of the sub-model as returned to the caller
class PartitionInvocationResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    context: PartitionInvocationContext

    payload: TensorBundle
