from typing import Self

from pydantic import BaseModel, ConfigDict, NonNegativeInt, model_validator

from shared.flow.flow import FlowId
from shared.plan.partition_replica_id import PartitionReplicaId


class PriorityKey(BaseModel):
    model_config = ConfigDict(frozen=True)

    flow_id: FlowId
    partition_replica_id: PartitionReplicaId


type PriorityValue = NonNegativeInt  ## Lower values are higher priority


class SchedulingPlan(BaseModel):
    model_config = ConfigDict(frozen=True)

    priority_keys: tuple[PriorityKey, ...]
    priorities: tuple[PriorityValue, ...]

    @model_validator(mode="after")
    def validate_scheduling_plan(self) -> Self:

        if len(self.priority_keys) != len(set(self.priority_keys)):
            raise ValueError("Priority keys must be unique")

        if len(self.priority_keys) == 0:
            raise ValueError("Scheduling plan must not be empty")

        if len(self.priority_keys) != len(self.priorities):
            raise ValueError("Priorities must be present for all priority keys")

        return self

    def get_priority_by_key(self, priority_key: PriorityKey) -> PriorityValue:
        return self.priorities[self.priority_keys.index(priority_key)]
