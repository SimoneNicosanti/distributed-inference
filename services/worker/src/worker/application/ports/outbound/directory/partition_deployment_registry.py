from abc import ABC, abstractmethod

from shared.plan.partition_deployment_id import PartitionDeploymentId
from shared.plan.partition_replica_id import PartitionReplicaId


class PartitionDeploymentRegistry(ABC):
    @abstractmethod
    async def register_partition_deployment(
        self, deployment_id: PartitionDeploymentId
    ) -> None: ...

    @abstractmethod
    async def get_partition_deployment_id(
        self, replica_id: PartitionReplicaId
    ) -> PartitionDeploymentId: ...
