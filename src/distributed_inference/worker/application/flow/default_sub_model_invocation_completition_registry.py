import asyncio
from typing import override

from distributed_inference.worker.application.flow.contracts.sub_model_invocation_completition_registry import (
    SubModelInvocationCompletitionRegistry,
)
from distributed_inference.worker.domain.sub_model.invocation.sub_model_invocation_context import (
    SubModelInvocationId,
)

## TODO asyncio.shield might be useful also for other things!


class DefaultCompletitionRegistry(SubModelInvocationCompletitionRegistry):
    def __init__(self) -> None:
        self._completition_events: dict[SubModelInvocationId, asyncio.Future[None]] = {}

    @override
    async def wait_for_sub_model_invocation_completition(
        self, sub_model_invocation_id: SubModelInvocationId
    ) -> None:
        if sub_model_invocation_id not in self._completition_events:
            future = asyncio.get_running_loop().create_future()
            self._completition_events[sub_model_invocation_id] = future
        await asyncio.shield(self._completition_events[sub_model_invocation_id])

    @override
    async def register_sub_model_invocation_success(
        self, sub_model_invocation_id: SubModelInvocationId
    ) -> None:
        if sub_model_invocation_id in self._completition_events:
            self._completition_events[sub_model_invocation_id].set_result(None)
        raise KeyError(f"Sub-model {sub_model_invocation_id} does not exist")

    @override
    async def register_sub_model_invocation_failure(
        self, sub_model_invocation_id: SubModelInvocationId, exception: BaseException
    ) -> None:
        if sub_model_invocation_id in self._completition_events:
            self._completition_events[sub_model_invocation_id].set_exception(exception)
        pass
