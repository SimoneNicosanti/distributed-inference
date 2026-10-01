from pydantic import BaseModel, ConfigDict

from shared.flow.flow import FlowId
from shared.model.sub_model import SubModelId
from shared.plan.plan import InferencePlanVersion, SubModelDeployment
from worker.domain.sub_model.invocation.sub_model_invocation_context import (
    SubModelInvocationContext,
)
from worker.domain.tensor.tensor import TensorBundle


## This is the input of the sub-model once everything for the sub-model has been gathered from predecessors
class SubModelInvocationRequest(BaseModel):
    model_config = ConfigDict(frozen=True)
    context: SubModelInvocationContext

    payload: TensorBundle

    @property
    def plan_version(self) -> InferencePlanVersion:
        return self.context.model_pass_context.plan_version

    @property
    def sub_model_id(self) -> SubModelId:
        return self.context.sub_model_deployment_id.sub_model_id

    @property
    def flow_id(self) -> FlowId:
        return self.context.model_pass_context.model_invocation_context.flow_id

    @property
    def sub_model_deployment_id(self) -> SubModelDeployment:
        return self.context.sub_model_deployment_id


## This is the output of the sub-model as returned to the caller
class SubModelInvocationResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    context: SubModelInvocationContext

    payload: TensorBundle
