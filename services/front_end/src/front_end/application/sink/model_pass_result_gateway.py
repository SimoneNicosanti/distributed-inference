from abc import ABC, abstractmethod

from shared.service.service import SinkId
from worker.domain.context.model_pass_context import ModelPassId
from worker.domain.model.model_pass import ModelPassResult


class ModelPassResultRetrieverGateway(ABC):
    @abstractmethod
    async def retrieve_result(
        self,
        request_id: ModelPassId,
        result_sink_id: SinkId,
    ) -> ModelPassResult: ...
