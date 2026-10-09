from pydantic import BaseModel, ConfigDict

from shared.context.model_pass_context import (
    ModelPassContext,
)
from shared.plan.partition_replica_id import PartitionReplicaId


class PartitionInvocationCollectionKey(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_pass_context: ModelPassContext
    partition_replica_id: PartitionReplicaId
