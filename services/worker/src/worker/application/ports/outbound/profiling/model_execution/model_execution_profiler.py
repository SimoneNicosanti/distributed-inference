from abc import ABC, abstractmethod

from artifacts.workspace.artifact_bundle import ArtifactBundle
from worker.domain.profiling.model_execution_profile import ModelExecutionProfile


class ModelExecutionProfiler(ABC):
    @abstractmethod
    async def profile_model_execution(
        self, artifact_bundle: ArtifactBundle
    ) -> ModelExecutionProfile: ...
