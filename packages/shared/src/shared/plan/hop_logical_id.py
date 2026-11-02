from dataclasses import dataclass

from shared.model.model_partition import ModelPartitionId
from shared.model.model_variant import ModelVariantId


## Identifier of a partition replica system-wide
## for a specific version of the plan
@dataclass(frozen=True, slots=True)
class PlannedPartitionReplicaId:
    partition_id: ModelPartitionId
    replica_idx: int

    def __post_init__(self) -> None:
        if isinstance(self.replica_idx, bool) or self.replica_idx < 0:
            raise ValueError("Replica index must be non-negative")


## Identifier of an ingress replica system-wide
## for a specific version of the plan
@dataclass(frozen=True, slots=True)
class PlannedIngressReplicaId:
    model_variant_id: ModelVariantId
    replica_idx: int

    def __post_init__(self) -> None:
        if isinstance(self.replica_idx, bool) or self.replica_idx < 0:
            raise ValueError("Replica index must be non-negative")


## Identifier of a sink replica system-wide
## for a specific version of the plan
@dataclass(frozen=True, slots=True)
class PlannedSinkReplicaId:
    model_variant_id: ModelVariantId
    replica_idx: int

    def __post_init__(self) -> None:
        if isinstance(self.replica_idx, bool) or self.replica_idx < 0:
            raise ValueError("Replica index must be non-negative")
