from abc import ABC, abstractmethod

from front_end.domain.context.model_pass_context import ModelPassContext
from front_end.domain.tensor_map import TensorMap


class SinkGateway(ABC):
    @abstractmethod
    async def retrieve_result(
        self, model_pass_context: ModelPassContext
    ) -> TensorMap: ...
