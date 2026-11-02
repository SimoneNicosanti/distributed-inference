from abc import ABC, abstractmethod

from shared.service.service import WorkerId
from worker.domain.hop.hop_invocation_contribution import HopInvocationContribution


class ContributionSender(ABC):
    @abstractmethod
    async def send(
        self,
        contribution: HopInvocationContribution,
        worker_id: WorkerId,
    ) -> None: ...
