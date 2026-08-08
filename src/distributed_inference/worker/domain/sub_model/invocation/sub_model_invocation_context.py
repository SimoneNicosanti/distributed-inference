## A model pass can be made up of multiple sub-models invocations.
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from distributed_inference.domain.plan import SubModelDeployment
from distributed_inference.worker.domain.model_pass.model_pass_context import (
    ModelPassContext,
)


class SubModelInvocationId(BaseModel):
    model_config = ConfigDict(frozen=True)
    id: UUID = Field(default_factory=uuid4)


## A model pass is made up of multiple sub-model invocations
## The context of a sub-model invocation is the identified by
## - The model pass it belongs to
## - The sub-model deployment it is invoking
## - The sub-model invocation id (unique within the sub-model deployment)
class SubModelInvocationContext(BaseModel):
    model_config = ConfigDict(frozen=True)
    model_pass_context: ModelPassContext
    sub_model_deployment_id: SubModelDeployment
    sub_model_invocation_id: SubModelInvocationId = Field(
        default_factory=SubModelInvocationId
    )
