from abc import ABC, abstractmethod

from distributed_inference.worker.domain.sub_model.invocation.sub_model_invocation_context import (
    SubModelInvocationId,
)


class SubModelInvocationCompletitionRegistry(ABC):
    @abstractmethod
    async def wait_for_sub_model_invocation_completition(
        self, sub_model_invocation_id: SubModelInvocationId
    ) -> None: ...

    @abstractmethod
    async def register_sub_model_invocation_success(
        self, sub_model_invocation_id: SubModelInvocationId
    ) -> None: ...

    @abstractmethod
    async def register_sub_model_invocation_failure(
        self, sub_model_invocation_id: SubModelInvocationId, exception: BaseException
    ) -> None: ...
