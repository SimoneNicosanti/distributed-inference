from pydantic import BaseModel, ConfigDict, NonNegativeInt

from shared.model.model_partition import ModelPartitionId


## Identifier of a partition replica system-wide
class PartitionReplicaId(BaseModel):
    model_config = ConfigDict(frozen=True)

    partition_id: ModelPartitionId
    replica_idx: NonNegativeInt
