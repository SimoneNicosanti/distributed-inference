## A model pass can be made up of multiple partition invocations.
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from shared.context.model_pass_context import (
    ModelPassContext,
)
from shared.plan.partition_replica_id import PartitionReplicaId

type PartitionInvocationId = UUID


## A model pass is made up of multiple partitions invocations
## The context of a partition invocation is the identified by
## - The model pass it belongs to
## - The partition replica it is invoking
## - The partition invocation id (unique for the partition replica)
class PartitionInvocationContext(BaseModel):
    model_config = ConfigDict(frozen=True)
    model_pass_context: ModelPassContext
    partition_replica_id: PartitionReplicaId
    partition_invocation_id: PartitionInvocationId = Field(default_factory=uuid4)
