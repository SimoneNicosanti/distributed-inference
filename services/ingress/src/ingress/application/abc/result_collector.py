from abc import ABC, abstractmethod

from shared.context.model_pass_context import ModelPassId
from shared.model.keys import TensorKey
from shared.tensor.tensor_bundle import TensorBundle


class ResultCollector(ABC):
    @abstractmethod
    async def collect_result(
        self,
        request_id: ModelPassId,
        to_wait_list: list[TensorKey],
    ) -> TensorBundle: ...

    @abstractmethod
    async def add_result_contribution(
        self,
        request_id: ModelPassId,
        result: TensorBundle,
    ) -> None: ...
