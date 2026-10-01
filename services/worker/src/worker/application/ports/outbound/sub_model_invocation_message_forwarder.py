from abc import ABC, abstractmethod

from worker.domain.sub_model.invocation.sub_model_invocation_message import (
    SubModelInvocationMessage,
)


class SubModelInvocationMessageForwarder(ABC):
    @abstractmethod
    def forward_sub_model_invocation_message(
        self,
        sub_model_invocation_message: SubModelInvocationMessage,
    ) -> None: ...
