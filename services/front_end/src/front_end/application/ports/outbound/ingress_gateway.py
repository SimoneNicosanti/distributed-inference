from abc import ABC, abstractmethod

from front_end.domain.context.external_source_context import (
    ExternalSourceInvocationContext,
)
from front_end.domain.tensor_map import TensorMap


class IngressGateway(ABC):
    @abstractmethod
    async def invoke_model(
        self,
        external_source_invocation_context: ExternalSourceInvocationContext,
        bundle: TensorMap,
    ) -> None: ...
