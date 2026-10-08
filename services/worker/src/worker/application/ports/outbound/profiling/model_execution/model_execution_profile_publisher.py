from abc import ABC, abstractmethod

from shared.service.service import WorkerId
from worker.domain.profiling.model_execution.model_execution_profile import (
    ModelExecutionProfile,
)


class ModelExecutionProfilePublisher(ABC):
    @abstractmethod
    async def puplish_model_execution_profile(
        self, worker_id: WorkerId, model_execution_profile: ModelExecutionProfile
    ) -> None: ...
