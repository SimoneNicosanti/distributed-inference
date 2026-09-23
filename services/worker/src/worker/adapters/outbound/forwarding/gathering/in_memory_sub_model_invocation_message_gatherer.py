from typing import override

from worker.application.forwarding.contracts.gathering.gather_key import (
    GatherKey,
)
from worker.application.ports.outbound.sub_model_invocation_message_store import (
    SubModelInvocationMessageGatheringStore,
)
from worker.domain.sub_model.invocation.sub_model_invocation_message import (
    SubModelInvocationMessage,
)


class InMemorySubModelInvocationMessageGatherer(
    SubModelInvocationMessageGatheringStore
):
    def __init__(self) -> None:
        super().__init__()
        self._memory_store: dict[GatherKey, list[SubModelInvocationMessage]] = {}

    @override
    async def put_sub_model_invocation_message(
        self, sub_model_invocation_message: SubModelInvocationMessage
    ) -> GatherKey:

        gather_key = GatherKey(
            model_pass_context=sub_model_invocation_message.model_pass_context,
            sub_model_deployment_id=sub_model_invocation_message.sub_model_deployment_id,
        )

        if gather_key not in self._memory_store:
            self._memory_store[gather_key] = []

        self._memory_store[gather_key].append(sub_model_invocation_message)

        return gather_key

    @override
    async def get_by_gathering_key(
        self, gathering_key: GatherKey
    ) -> list[SubModelInvocationMessage]:

        return self._memory_store[gathering_key]

    @override
    async def delete_by_gathering_key(self, gathering_key: GatherKey) -> None:
        self._memory_store.pop(gathering_key)
