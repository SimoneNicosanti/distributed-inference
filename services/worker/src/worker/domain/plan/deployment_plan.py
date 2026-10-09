from typing import Self

from pydantic import BaseModel, ConfigDict, model_validator

from shared.plan.partition_replica_id import PartitionReplicaId


class ResourceAllocation(BaseModel):
    model_config = ConfigDict(frozen=True)

    use_gpu: bool


class DeploymentPlan(BaseModel):
    model_config = ConfigDict(frozen=True)

    replica_ids: tuple[PartitionReplicaId, ...]
    allocations: tuple[ResourceAllocation, ...]

    @model_validator(mode="after")
    def validate_deployments(self) -> Self:

        if len(self.replica_ids) == 0:
            raise ValueError("Deployments must not be empty")

        if len(self.replica_ids) != len(set(self.replica_ids)):
            raise ValueError("Replicas must be unique")

        if len(self.replica_ids) != len(self.allocations):
            raise ValueError("Each replica must have an allocation")

        return self

    def get_allocation_by_replica_id(
        self, replica_id: PartitionReplicaId
    ) -> ResourceAllocation:
        return self.allocations[self.replica_ids.index(replica_id)]
