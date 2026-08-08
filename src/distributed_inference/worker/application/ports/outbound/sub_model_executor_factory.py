from abc import ABC, abstractmethod

from distributed_inference.artifact_materializer.domain.materialized_artifact import (
    MaterializedArtifact,
)
from distributed_inference.domain.plan import ResourceAllocation
from distributed_inference.worker.application.ports.outbound.sub_model_executor import (
    SubModelExecutor,
)


class SubModelExecutorFactory(ABC):
    @abstractmethod
    async def create_sub_model_executor(
        self,
        materialized_artifact: MaterializedArtifact,
        resource_allocation: ResourceAllocation,
    ) -> SubModelExecutor: ...
