from typing import Self

from pydantic import BaseModel, ConfigDict, model_validator

from shared.model.keys import TensorKey
from shared.model.model_partition import ModelPartitionId


class PartitionIOScheme(BaseModel):
    model_config = ConfigDict(frozen=True)

    inputs: tuple[TensorKey, ...]
    outputs: tuple[TensorKey, ...]


class PartitionExecutionPlan(BaseModel):
    model_config = ConfigDict(frozen=True)

    partition_ids: tuple[ModelPartitionId, ...]
    schemes: tuple[PartitionIOScheme, ...]

    @model_validator(mode="after")
    def validate_execution_plan(self) -> Self:

        if len(self.partition_ids) == 0:
            raise ValueError("Execution plan must not be empty")

        if len(self.partition_ids) != len(set(self.partition_ids)):
            raise ValueError("Partitions must be unique")

        if len(self.partition_ids) != len(self.schemes):
            raise ValueError("Each partition must have an execution scheme")

        return self

    def get_scheme_by_partition_id(
        self, partition_id: ModelPartitionId
    ) -> PartitionIOScheme:
        return self.schemes[self.partition_ids.index(partition_id)]
