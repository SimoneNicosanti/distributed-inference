from abc import ABC, abstractmethod

from artifacts.contracts.artifact_bundle import ArtifactBundle
from shared.plan.plan import ResourceAllocation
from worker.application.ports.outbound.partition_execution.partition_executor import (
    PartitionExecutor,
)


class PartitionExecutorFactory(ABC):
    @abstractmethod
    async def create(
        self,
        bundle: ArtifactBundle,
        resource_allocation: ResourceAllocation,
    ) -> PartitionExecutor: ...
