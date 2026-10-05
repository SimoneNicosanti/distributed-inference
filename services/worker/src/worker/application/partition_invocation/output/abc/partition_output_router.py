from abc import ABC, abstractmethod

from worker.domain.partition.partition_invocation import PartitionInvocationResult
from worker.domain.partition.partition_invocation_contribution import (
    PartitionInvocationContribution,
)


class PartitionOutputRouter(ABC):
    @abstractmethod
    async def route(
        self,
        contributions: list[PartitionInvocationContribution],
        partition_invocation_result: PartitionInvocationResult,
    ) -> list[PartitionInvocationContribution]: ...
