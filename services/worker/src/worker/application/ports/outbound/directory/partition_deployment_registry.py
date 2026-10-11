from abc import ABC, abstractmethod

from shared.plan.partition_deployment_id import PartitionDeploymentId
from shared.plan.partition_replica_id import PartitionReplicaId
from shared.plan.plan_version import PlanVersion


class PartitionDeploymentRegistry(ABC):
    @abstractmethod
    async def register_partition_deployment(
        self,
        plan_version: PlanVersion,
        deployment_id: PartitionDeploymentId,
    ) -> None: ...

    @abstractmethod
    async def get_partition_deployment_id(
        self,
        plan_version: PlanVersion,
        replica_id: PartitionReplicaId,
    ) -> PartitionDeploymentId: ...
