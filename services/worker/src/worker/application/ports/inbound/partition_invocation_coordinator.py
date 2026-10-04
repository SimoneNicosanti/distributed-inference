from abc import ABC, abstractmethod

from worker.domain.partition.partition_invocation_contribution import (
    PartitionInvocationContribution,
    PartitionInvocationContributionAck,
)


## Coordinates a partition invocation from contribution reception through routing.
class PartitionInvocationCoordinator(ABC):
    @abstractmethod
    async def process_partition_invocation_contribution(
        self, contribution: PartitionInvocationContribution
    ) -> PartitionInvocationContributionAck: ...
