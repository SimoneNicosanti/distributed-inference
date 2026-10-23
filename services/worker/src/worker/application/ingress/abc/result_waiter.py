from abc import ABC, abstractmethod
from asyncio import Future

from shared.context.model_pass_context import ModelPassId
from shared.model.keys import TensorKey
from worker.domain.tensor.tensor_bundle import TensorBundle


class ResultWaiter(ABC):
    @abstractmethod
    async def register_result_wait(
        self,
        request_id: ModelPassId,
        to_wait_list: list[TensorKey],
    ) -> Future[TensorBundle]: ...
