from abc import ABC, abstractmethod

from shared.plan.hop_deployment_id import HopDeploymentId, HopReplicaId
from shared.plan.plan_version import PlanVersion


class HopDeploymentDirectory(ABC):
    @abstractmethod
    async def get_hop_deployment_id[ReplicaIdT: HopReplicaId](
        self,
        plan_version: PlanVersion,
        replica_id: ReplicaIdT,
    ) -> HopDeploymentId[ReplicaIdT]: ...
