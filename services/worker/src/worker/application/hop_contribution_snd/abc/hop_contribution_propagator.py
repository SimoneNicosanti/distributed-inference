from abc import ABC, abstractmethod

from worker.domain.hop.hop import InvocableHop
from worker.domain.hop.hop_invocation import HopInvocationResult
from worker.domain.hop.hop_invocation_contribution import HopInvocationContribution


class HopContributionPropagator(ABC):
    @abstractmethod
    async def propagate(
        self,
        incoming_contributions: list[HopInvocationContribution],
        hop_invocation_result: HopInvocationResult[InvocableHop],
    ) -> None: ...
