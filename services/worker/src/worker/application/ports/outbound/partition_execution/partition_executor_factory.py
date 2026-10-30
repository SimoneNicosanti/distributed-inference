from abc import ABC, abstractmethod

from artifacts.contracts.artifact_bundle import ArtifactBundle
from worker.application.ports.outbound.partition_execution.partition_executor import (
    PartitionExecutor,
)
from worker.domain.plan.deployment_plan import ResourceAllocation


class PartitionExecutorFactory(ABC):
    @abstractmethod
    async def create(
        self,
        bundle: ArtifactBundle,
        resource_allocation: ResourceAllocation,
    ) -> PartitionExecutor: ...
