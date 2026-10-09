from typing import override

from worker.application.partition_invocation.input.partition_invocation_collection_key import (
    PartitionInvocationCollectionKey,
)
from worker.application.ports.outbound.partition_invocation.store.partition_invocation_contribution_store import (
    PartitionInvocationContributionStore,
)
from worker.domain.partition.partition_invocation_contribution import (
    PartitionInvocationContribution,
)


class InMemoryPartitionInvocationContributionStore(
    PartitionInvocationContributionStore
):
    def __init__(self) -> None:
        super().__init__()
        self._memory_store: dict[
            PartitionInvocationCollectionKey, list[PartitionInvocationContribution]
        ] = {}

    @override
    async def put(
        self, contribution: PartitionInvocationContribution
    ) -> PartitionInvocationCollectionKey:

        collection_key = PartitionInvocationCollectionKey(
            model_pass_context=contribution.model_pass_context,
            partition_replica_id=contribution.target_replica_id,
        )

        if collection_key not in self._memory_store:
            self._memory_store[collection_key] = []

        self._memory_store[collection_key].append(contribution)

        return collection_key

    @override
    async def get(
        self, collection_key: PartitionInvocationCollectionKey
    ) -> list[PartitionInvocationContribution]:

        return self._memory_store[collection_key]

    @override
    async def pop(
        self, collection_key: PartitionInvocationCollectionKey
    ) -> list[PartitionInvocationContribution]:
        return self._memory_store.pop(collection_key)
