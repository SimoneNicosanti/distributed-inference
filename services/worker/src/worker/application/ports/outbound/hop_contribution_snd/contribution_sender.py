from abc import ABC, abstractmethod

from worker.domain.hop.hop_invocation_contribution import HopInvocationContribution


class ContributionSender(ABC):
    @abstractmethod
    async def send(
        self,
        contribution: HopInvocationContribution,
    ) -> None: ...
