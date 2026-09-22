from abc import ABC, abstractmethod

from worker.domain.sub_model.invocation.sub_model_invocation_request_response import (
    SubModelInvocationResponse,
)
from worker.domain.sub_model.route.sub_model_invocation_response_route import (
    SubModelInvocationResponseRoute,
)


class SubModelInvocationResponseRouter(ABC):
    @abstractmethod
    async def route_sub_model_invocation_response(
        self, sub_model_inference_response: SubModelInvocationResponse
    ) -> list[SubModelInvocationResponseRoute]: ...
