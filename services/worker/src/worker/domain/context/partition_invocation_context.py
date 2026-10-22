## A model pass can be made up of multiple partition invocations.
from dataclasses import dataclass, field
from uuid import UUID, uuid4

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
@dataclass(frozen=True, slots=True)
class PartitionInvocationContext:
    model_pass_context: ModelPassContext
    partition_replica_id: PartitionReplicaId
    partition_invocation_id: PartitionInvocationId = field(default_factory=uuid4)
