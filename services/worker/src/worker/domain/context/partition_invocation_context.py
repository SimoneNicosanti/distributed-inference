## A model pass can be made up of multiple partition invocations.
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from shared.plan.plan import PartitionDeployment
from worker.domain.context.model_pass_context import (
    ModelPassContext,
)

type PartitionInvocationId = UUID


## A model pass is made up of multiple partitions invocations
## The context of a partition invocation is the identified by
## - The model pass it belongs to
## - The partition deployment it is invoking
## - The partition invocation id (unique for the partition deployment)
class PartitionInvocationContext(BaseModel):
    model_config = ConfigDict(frozen=True)
    model_pass_context: ModelPassContext
    partition_deployment_id: PartitionDeployment
    partition_invocation_id: PartitionInvocationId = Field(default_factory=uuid4)
