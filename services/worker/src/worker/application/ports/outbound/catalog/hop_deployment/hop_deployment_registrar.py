from abc import ABC, abstractmethod

from shared.plan.hop_deployment_id import HopDeploymentId, HopReplicaId
from shared.plan.plan_version import PlanVersion


class HopDeploymentRegistrar(ABC):
    @abstractmethod
    async def register_hop_deployments[ReplicaIdT: HopReplicaId](
        self,
        plan_version: PlanVersion,
        deployment_ids: tuple[HopDeploymentId[ReplicaIdT], ...],
    ) -> None: ...
