from abc import ABC, abstractmethod

from shared.context.model_pass_context import ModelPassId
from worker.domain.model_pass.model_pass_result import ModelPassResult


class ModelPassResultWaiter(ABC):
    @abstractmethod
    async def wait_result(
        self,
        request_id: ModelPassId,
    ) -> ModelPassResult: ...
