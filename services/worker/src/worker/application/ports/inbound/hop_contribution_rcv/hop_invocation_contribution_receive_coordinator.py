from abc import ABC, abstractmethod

from worker.domain.hop.hop_invocation_contribution import (
    HopInvocationContribution,
    HopInvocationContributionAck,
)


class HopInvocationContributionReceiveCoordinator(ABC):
    @abstractmethod
    async def pass_hop_invocation_contribution(
        self, contribution: HopInvocationContribution
    ) -> HopInvocationContributionAck: ...
