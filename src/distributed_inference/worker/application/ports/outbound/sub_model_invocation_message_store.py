from abc import ABC, abstractmethod

from distributed_inference.worker.domain.sub_model.invocation.sub_model_invocation_message import (
    SubModelInvocationMessage,
    SubModelInvocationMessageContext,
)


class SubModelInvocationMessageStore(ABC):
    @abstractmethod
    async def put_sub_model_invocation_message(
        self, sub_model_invocation_message: SubModelInvocationMessage
    ) -> None: ...

    @abstractmethod
    async def get_sub_model_invocation_message_by_context(
        self, sub_model_invocation_message_context: SubModelInvocationMessageContext
    ) -> SubModelInvocationMessage: ...
