from abc import ABC, abstractmethod

from shared.plan.hop import PlannedDestinationHopId
from worker.domain.hop.hop_invocation import HopInvocationRequest
from worker.domain.hop.hop_invocation_contribution import HopInvocationContribution


class ContributionAssemblingError(Exception):
    pass


class ContributionAssembler(ABC):
    @abstractmethod
    async def assemble(
        self, contributions: list[HopInvocationContribution]
    ) -> HopInvocationRequest[PlannedDestinationHopId]: ...
