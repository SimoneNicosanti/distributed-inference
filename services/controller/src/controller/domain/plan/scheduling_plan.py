from dataclasses import dataclass

from shared.flow.flow import FlowId
from shared.plan.partition_replica_id import PartitionReplicaId


@dataclass(frozen=True, slots=True)
class PriorityKey:
    flow_id: FlowId
    partition_replica_id: PartitionReplicaId


type PriorityValue = int


@dataclass(frozen=True, slots=True)
class SchedulingPlan:
    priority_keys: tuple[PriorityKey, ...]
    priorities: tuple[PriorityValue, ...]

    def __post_init__(self) -> None:
        if len(self.priority_keys) != len(set(self.priority_keys)):
            raise ValueError("Priority keys must be unique")
        if len(self.priority_keys) != len(self.priorities):
            raise ValueError("Priorities must be present for all priority keys")
        if any(
            isinstance(priority, bool) or priority < 0 for priority in self.priorities
        ):
            raise ValueError("Priorities must be non-negative integers")
