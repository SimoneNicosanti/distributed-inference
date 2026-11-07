from abc import ABC, abstractmethod

from front_end.domain.tensor_map import TensorMap
from shared.context.model_pass_context import ModelPassContext
from shared.service.service import WorkerId


class ModelPassFailedError(Exception):
    """The sink returned a negative model pass outcome."""


class SinkGateway(ABC):
    @abstractmethod
    async def retrieve_result(
        self,
        model_pass_context: ModelPassContext,
        worker_id: WorkerId,
    ) -> TensorMap: ...
