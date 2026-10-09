from abc import ABC, abstractmethod

from worker.domain.model_pass.model_pass_result_contribution import (
    ModelPassResultContribution,
)
from worker.domain.partition.partition_invocation_contribution import (
    PartitionInvocationContribution,
)


class ContributionSendCoordinator(ABC):
    @abstractmethod
    async def send_all_contributions(
        self,
        all_contributions: list[
            PartitionInvocationContribution | ModelPassResultContribution
        ],
    ) -> None: ...
