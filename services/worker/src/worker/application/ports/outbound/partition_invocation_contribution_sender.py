from abc import ABC, abstractmethod

from worker.domain.partition.partition_invocation_contribution import (
    PartitionInvocationContribution,
)


class PartitionInvocationContributionSender(ABC):
    @abstractmethod
    def send(
        self,
        contribution: PartitionInvocationContribution,
    ) -> None: ...
