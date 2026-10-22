from dataclasses import dataclass

from shared.model.model_partition import ModelPartitionId


## Identifier of a partition replica system-wide
@dataclass(frozen=True, slots=True)
class PartitionReplicaId:
    partition_id: ModelPartitionId
    replica_idx: int

    def __post_init__(self) -> None:
        if isinstance(self.replica_idx, bool) or self.replica_idx < 0:
            raise ValueError("Replica index must be non-negative")
