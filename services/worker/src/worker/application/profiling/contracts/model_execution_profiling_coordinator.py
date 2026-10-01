from abc import ABC, abstractmethod

from shared.model.model_version import ModelVersionId
from worker.domain.profiling.model_execution_profile import ModelExecutionProfile


class ModelExecutionProfilingCoordinator(ABC):
    @abstractmethod
    async def profile_model_execution(
        self,
        model_version_id: ModelVersionId,
    ) -> ModelExecutionProfile: ...
