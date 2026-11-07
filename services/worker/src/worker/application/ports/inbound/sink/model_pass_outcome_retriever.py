from abc import ABC, abstractmethod

from shared.context.model_pass_context import ModelPassId
from worker.domain.model.model_pass import ModelPassOutcome


class ModelPassOutcomeRetriever(ABC):
    @abstractmethod
    async def retrieve_outcome(
        self,
        request_id: ModelPassId,
    ) -> ModelPassOutcome: ...
