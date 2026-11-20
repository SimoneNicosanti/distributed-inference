from abc import ABC, abstractmethod

from artifacts.contracts.artifact_bundle import ArtifactBundle
from worker.application.ports.outbound.hop_execution.hop_executor import (
    HopExecutor,
)
from worker.domain.plan.deployment_plan import ResourceAllocation


class PartitionExecutorCreationError(Exception):
    pass


class HopExecutorFactory(ABC):
    @abstractmethod
    async def create(
        self,
        bundle: ArtifactBundle,
        resource_allocation: ResourceAllocation,
    ) -> HopExecutor: ...
