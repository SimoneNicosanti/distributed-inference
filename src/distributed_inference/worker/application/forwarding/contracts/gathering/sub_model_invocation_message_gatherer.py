from abc import ABC, abstractmethod
from typing import Tuple

from distributed_inference.worker.domain.sub_model.invocation.sub_model_invocation_context import (
    SubModelInvocationId,
)
from distributed_inference.worker.domain.sub_model.invocation.sub_model_invocation_message import (
    SubModelInvocationMessage,
)
from distributed_inference.worker.domain.sub_model.invocation.sub_model_invocation_request_response import (
    SubModelInvocationRequest,
)


class SubModelInvocationMessageGatherer(ABC):
    @abstractmethod
    async def gather_sub_model_invocation_message(
        self, sub_model_invocation_message: SubModelInvocationMessage
    ) -> Tuple[SubModelInvocationRequest | None, SubModelInvocationId]: ...
