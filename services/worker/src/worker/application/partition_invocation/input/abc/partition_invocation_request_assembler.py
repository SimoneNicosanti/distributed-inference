from abc import ABC, abstractmethod

from worker.domain.partition.partition_invocation import PartitionInvocationRequest
from worker.domain.partition.partition_invocation_contribution import (
    PartitionInvocationContribution,
)


class PartitionInvocationRequestAssembler(ABC):
    @abstractmethod
    async def assemble(
        self,
        contributions: list[PartitionInvocationContribution],
    ) -> PartitionInvocationRequest: ...
