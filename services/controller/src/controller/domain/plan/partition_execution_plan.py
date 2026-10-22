from dataclasses import dataclass

from shared.model.keys import TensorKey
from shared.model.model_partition import ModelPartitionId


@dataclass(frozen=True, slots=True)
class PartitionIOScheme:
    inputs: tuple[TensorKey, ...]
    outputs: tuple[TensorKey, ...]


@dataclass(frozen=True, slots=True)
class PartitionExecutionPlan:
    partition_ids: tuple[ModelPartitionId, ...]
    schemes: tuple[PartitionIOScheme, ...]

    def __post_init__(self) -> None:
        if len(self.partition_ids) != len(set(self.partition_ids)):
            raise ValueError("Partitions must be unique")

        if len(self.partition_ids) != len(self.schemes):
            raise ValueError("Each partition must have an execution scheme")
