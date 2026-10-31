from dataclasses import dataclass

from shared.model.model_partition import ModelPartitionId
from shared.model.model_variant import ModelVariantId


## Identifier of a partition replica system-wide
@dataclass(frozen=True, slots=True)
class PartitionReplicaId:
    partition_id: ModelPartitionId
    replica_idx: int

    def __post_init__(self) -> None:
        if isinstance(self.replica_idx, bool) or self.replica_idx < 0:
            raise ValueError("Replica index must be non-negative")


## Identifier of an ingress replica system-wide
@dataclass(frozen=True, slots=True)
class IngressReplicaId:
    model_variant_id: ModelVariantId
    replica_idx: int

    def __post_init__(self) -> None:
        if isinstance(self.replica_idx, bool) or self.replica_idx < 0:
            raise ValueError("Replica index must be non-negative")


## Identifier of a sink replica system-wide
@dataclass(frozen=True, slots=True)
class SinkReplicaId:
    model_variant_id: ModelVariantId
    replica_idx: int

    def __post_init__(self) -> None:
        if isinstance(self.replica_idx, bool) or self.replica_idx < 0:
            raise ValueError("Replica index must be non-negative")
