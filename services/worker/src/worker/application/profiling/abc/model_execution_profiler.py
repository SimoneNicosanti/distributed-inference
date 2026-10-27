from abc import ABC, abstractmethod

from artifacts.contracts.artifact_bundle import ArtifactBundle
from worker.domain.profiling.model_execution.model_execution_profile import (
    ModelExecutionProfile,
)
from worker.domain.profiling.model_execution.model_static_profile import (
    ModelStaticProfile,
)


class ModelExecutionProfiler(ABC):
    @abstractmethod
    async def profile_model_execution(
        self, model_static_profile: ModelStaticProfile, artifact_bundle: ArtifactBundle
    ) -> ModelExecutionProfile: ...
