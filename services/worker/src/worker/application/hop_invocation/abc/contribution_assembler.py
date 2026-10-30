from abc import ABC, abstractmethod

from worker.domain.hop.hop import DestinationHop
from worker.domain.hop.hop_invocation import HopInvocationRequest
from worker.domain.hop.hop_invocation_contribution import HopInvocationContribution


class ContributionAssembler(ABC):
    @abstractmethod
    async def assemble(
        self, contributions: list[HopInvocationContribution]
    ) -> HopInvocationRequest[DestinationHop]: ...
