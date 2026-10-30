from abc import ABC, abstractmethod

from worker.domain.context.model_pass_context import ModelPassId
from worker.domain.model.model_pass import ModelPassResult


class ModelPassResultRetriever(ABC):
    @abstractmethod
    async def retrieve_result(
        self,
        request_id: ModelPassId,
    ) -> ModelPassResult: ...
