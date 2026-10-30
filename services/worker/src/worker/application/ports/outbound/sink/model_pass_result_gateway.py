from abc import ABC, abstractmethod

from worker.domain.context.model_pass_context import ModelPassId
from shared.service.service import SinkId
from worker.domain.model_pass.model_pass_result import ModelPassResult


class ModelPassResultRetrieverGateway(ABC):
    @abstractmethod
    async def retrieve_result(
        self,
        request_id: ModelPassId,
        result_sink_id: SinkId,
    ) -> ModelPassResult: ...
