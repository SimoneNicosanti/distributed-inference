from pydantic import BaseModel, ConfigDict

from distributed_inference.domain.plan import InferencePlanVersion
from distributed_inference.worker.domain.sub_model.invocation.sub_model_invocation_context import (
    SubModelInvocationContext,
)
from distributed_inference.worker.domain.tensor.tensor import TensorBundle


## This is the input of the sub-model once everything for the sub-model has been gathered from predecessors
class SubModelInvocationRequest(BaseModel):
    model_config = ConfigDict(frozen=True)
    context: SubModelInvocationContext

    payload: TensorBundle

    def get_plan_version(self) -> InferencePlanVersion:
        return self.context.model_pass_context.plan_version


## This is the output of the sub-model as returned to the caller
class SubModelInvocationResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    context: SubModelInvocationContext

    payload: TensorBundle
