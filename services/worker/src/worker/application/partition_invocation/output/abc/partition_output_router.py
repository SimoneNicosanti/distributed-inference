from abc import ABC, abstractmethod

from worker.domain.model_pass.model_pass_result_contribution import (
    ModelPassResultContribution,
)
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
    ) -> list[PartitionInvocationContribution | ModelPassResultContribution]: ...
