from typing import List, override

from distributed_inference.worker.application.forwarding.contracts.gathering.gather_key import (
    GatherKey,
)
from distributed_inference.worker.application.ports.outbound.sub_model_invocation_message_store import (
    SubModelInvocationMessageGatheringStore,
)
from distributed_inference.worker.domain.sub_model.invocation.sub_model_invocation_message import (
    SubModelInvocationMessage,
)


class InMemorySubModelInvocationMessageGatherer(
    SubModelInvocationMessageGatheringStore
):
    def __init__(self) -> None:
        super().__init__()
        self._memory_store: dict[GatherKey, List[SubModelInvocationMessage]] = {}

    @override
    async def put_sub_model_invocation_message(
        self, sub_model_invocation_message: SubModelInvocationMessage
    ) -> GatherKey:

        gather_key = GatherKey(
            model_pass_context=sub_model_invocation_message.context.model_pass_context,
            sub_model_deployment_id=sub_model_invocation_message.context.sub_model_deployment_id,
        )

        if gather_key not in self._memory_store:
            self._memory_store[gather_key] = []

        self._memory_store[gather_key].append(sub_model_invocation_message)

        return gather_key

    @override
    async def get_all_sub_model_invocation_message_by_gathering_key(
        self, gathering_key: GatherKey
    ) -> List[SubModelInvocationMessage]:

        return self._memory_store[gathering_key]
