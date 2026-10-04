from abc import ABC, abstractmethod

from worker.domain.context.partition_invocation_context import (
    PartitionInvocationId,
)
from worker.domain.partition.partition_invocation_contribution import (
    PartitionInvocationContribution,
)


class PartitionInvocationContributionCollector(ABC):
    @abstractmethod
    async def collect(
        self, contribution: PartitionInvocationContribution
    ) -> tuple[list[PartitionInvocationContribution] | None, PartitionInvocationId]: ...
