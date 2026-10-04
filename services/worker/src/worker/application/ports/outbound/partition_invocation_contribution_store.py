from abc import ABC, abstractmethod

from worker.application.partition_invocation.input.partition_invocation_collection_key import (
    PartitionInvocationCollectionKey,
)
from worker.domain.partition.partition_invocation_contribution import (
    PartitionInvocationContribution,
)


class PartitionInvocationContributionStore(ABC):
    @abstractmethod
    async def put(
        self, contribution: PartitionInvocationContribution
    ) -> PartitionInvocationCollectionKey: ...

    @abstractmethod
    async def get(
        self, collection_key: PartitionInvocationCollectionKey
    ) -> list[PartitionInvocationContribution]: ...

    @abstractmethod
    async def delete(
        self, collection_key: PartitionInvocationCollectionKey
    ) -> None: ...
