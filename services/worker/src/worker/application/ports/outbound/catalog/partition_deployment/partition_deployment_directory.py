from abc import ABC, abstractmethod

from shared.plan.partition_deployment_id import PartitionDeploymentId
from shared.plan.partition_replica_id import PartitionReplicaId
from shared.plan.plan_version import PlanVersion


class PartitionDeploymentDirectory(ABC):
    @abstractmethod
    async def get_partition_deployment_id(
        self,
        plan_version: PlanVersion,
        replica_id: PartitionReplicaId,
    ) -> PartitionDeploymentId: ...
