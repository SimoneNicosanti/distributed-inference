from abc import ABC, abstractmethod
from asyncio import Future

from worker.application.partition_invocation.input.ready_partition_invocation_request import (
    ReadyPartitionInvocationRequest,
)
from worker.domain.partition.partition_invocation_contribution import (
    PartitionInvocationContribution,
)


class PartitionInvocationContributionInbox(ABC):
    @abstractmethod
    async def submit_contribution(
        self, contribution: PartitionInvocationContribution
    ) -> Future[None]: ...

    @abstractmethod
    async def next_ready(
        self,
    ) -> ReadyPartitionInvocationRequest: ...
