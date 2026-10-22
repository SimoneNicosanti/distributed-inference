from abc import ABC, abstractmethod

from shared.plan.partition_deployment_id import PartitionDeploymentId
from shared.plan.plan_version import PlanVersion


class PartitionDeploymentRegistrar(ABC):
    @abstractmethod
    async def register_partition_deployment(
        self,
        plan_version: PlanVersion,
        deployment_id: PartitionDeploymentId,
    ) -> None: ...
