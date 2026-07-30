from abc import ABC, abstractmethod

from distributed_inference.worker.domain.sub_model.invocation.sub_model_invocation_message import (
    SubModelInvocationMessage,
)
from distributed_inference.worker.domain.sub_model.route.sub_model_invocation_response_route import (
    SubModelInvocationResponseRoute,
)


class SubModelInvocationMessageForwarder(ABC):
    @abstractmethod
    def forward_sub_model_invocation_message(
        self,
        sub_model_invocation_message: SubModelInvocationMessage,
        sub_model_invocation_response_route: SubModelInvocationResponseRoute,
    ) -> None: ...
