from abc import ABC, abstractmethod

from worker.domain.hop.hop_invocation_contribution import (
    HopInvocationContribution,
)


class HopContributionDispatcher(ABC):
    @abstractmethod
    async def dispatch(
        self,
        contributions: list[HopInvocationContribution],
    ) -> None: ...
