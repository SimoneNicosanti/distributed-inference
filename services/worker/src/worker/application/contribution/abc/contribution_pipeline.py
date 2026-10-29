import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CollectionHandle:
    future: asyncio.Future[None]


class ContributionPipeline[ConT](ABC):
    @abstractmethod
    async def submit(self, contribution: ConT) -> CollectionHandle: ...
