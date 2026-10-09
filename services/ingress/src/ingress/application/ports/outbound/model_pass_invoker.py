from abc import ABC, abstractmethod

from ingress.domain.model_pass_request import ModelPassRequest


class ModelPassInvoker(ABC):
    @abstractmethod
    async def invoke(
        self,
        request: ModelPassRequest,
    ) -> None: ...
