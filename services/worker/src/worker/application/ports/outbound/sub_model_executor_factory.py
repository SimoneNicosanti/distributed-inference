from abc import ABC, abstractmethod

from artifacts.workspace.artifact_bundle import ArtifactBundle
from shared.plan.plan import ResourceAllocation
from worker.application.ports.outbound.sub_model_executor import (
    SubModelExecutor,
)


class SubModelExecutorFactory(ABC):
    @abstractmethod
    async def create_sub_model_executor(
        self,
        bundle: ArtifactBundle,
        resource_allocation: ResourceAllocation,
    ) -> SubModelExecutor: ...
