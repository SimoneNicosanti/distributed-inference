from dataclasses import dataclass

from shared.context.model_pass_context import (
    ModelPassContext,
)
from shared.plan.partition_replica_id import PartitionReplicaId


@dataclass(frozen=True, slots=True)
class PartitionInvocationCollectionKey:
    model_pass_context: ModelPassContext
    partition_replica_id: PartitionReplicaId
