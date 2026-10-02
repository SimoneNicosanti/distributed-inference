from abc import ABC, abstractmethod

from artifacts.workspace.artifact_bundle import ArtifactBundle
from shared.model.model_version import ModelVersionId
from worker.domain.profiling.model_execution_profile import ModelExecutionProfile


class ModelExecutionProfiler(ABC):
    @abstractmethod
    async def profile_model_execution(
        self, model_version_id: ModelVersionId, artifact_bundle: ArtifactBundle
    ) -> ModelExecutionProfile: ...
