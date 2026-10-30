from abc import ABC, abstractmethod

from worker.domain.hop.hop import InvocableHop
from worker.domain.hop.hop_invocation import HopInvocationResult
from worker.domain.hop.hop_invocation_contribution import (
    HopInvocationContribution,
)


class HopOutputRouter(ABC):
    @abstractmethod
    async def route(
        self,
        incoming_contributions: list[HopInvocationContribution],
        hop_invocation_result: HopInvocationResult[InvocableHop],
    ) -> list[HopInvocationContribution]: ...
