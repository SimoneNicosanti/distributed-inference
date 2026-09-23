from abc import ABC, abstractmethod

from worker.domain.sub_model.invocation.sub_model_invocation_context import (
    SubModelInvocationId,
)
from worker.domain.sub_model.invocation.sub_model_invocation_message import (
    SubModelInvocationMessage,
)


class SubModelInvocationMessageGatherer(ABC):
    @abstractmethod
    async def gather_sub_model_invocation_message(
        self, sub_model_invocation_message: SubModelInvocationMessage
    ) -> tuple[list[SubModelInvocationMessage] | None, SubModelInvocationId]: ...
