from dataclasses import dataclass

from shared.plan.partition_replica_id import PartitionReplicaId


@dataclass(frozen=True, slots=True)
class ResourceAllocation:
    use_gpu: bool


@dataclass(frozen=True, slots=True)
class DeploymentPlan:
    replica_ids: tuple[PartitionReplicaId, ...]
    allocations: tuple[ResourceAllocation, ...]

    def __post_init__(self) -> None:
        if len(self.replica_ids) != len(set(self.replica_ids)):
            raise ValueError("Replicas must be unique")

        if len(self.replica_ids) != len(self.allocations):
            raise ValueError("Each replica must have an allocation")

    def get_allocation_by_replica_id(
        self, replica_id: PartitionReplicaId
    ) -> ResourceAllocation:
        return self.allocations[self.replica_ids.index(replica_id)]
