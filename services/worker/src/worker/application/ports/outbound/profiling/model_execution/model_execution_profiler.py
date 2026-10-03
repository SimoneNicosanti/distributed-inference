from abc import ABC, abstractmethod

from artifacts.contracts.artifact_bundle import ArtifactBundle
from shared.model.model_variant import ModelVariantId
from worker.domain.profiling.model_execution_profile import ModelExecutionProfile


class ModelExecutionProfiler(ABC):
    @abstractmethod
    async def profile_model_execution(
        self, model_version_id: ModelVariantId, artifact_bundle: ArtifactBundle
    ) -> ModelExecutionProfile: ...
