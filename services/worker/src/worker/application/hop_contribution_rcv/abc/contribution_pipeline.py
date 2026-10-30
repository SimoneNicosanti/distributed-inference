import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass

from worker.domain.hop.hop_invocation_contribution import HopInvocationContribution


@dataclass(frozen=True, slots=True)
class CollectionHandle:
    future: asyncio.Future[None]


class ContributionPipeline(ABC):
    @abstractmethod
    async def submit(
        self, contribution: HopInvocationContribution
    ) -> CollectionHandle: ...
