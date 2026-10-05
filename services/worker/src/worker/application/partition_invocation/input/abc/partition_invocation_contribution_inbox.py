import asyncio
from abc import ABC, abstractmethod
from asyncio import Future
from dataclasses import dataclass

from worker.domain.partition.partition_invocation import PartitionInvocationRequest
from worker.domain.partition.partition_invocation_contribution import (
    PartitionInvocationContribution,
)


@dataclass
class ReadyPartitionInvocationRequest:
    request: PartitionInvocationRequest
    contributions: list[PartitionInvocationContribution]
    future: asyncio.Future[None]


class PartitionInvocationContributionInbox(ABC):
    @abstractmethod
    async def submit_contribution(
        self, contribution: PartitionInvocationContribution
    ) -> Future[None]: ...

    @abstractmethod
    async def next_ready(
        self,
    ) -> ReadyPartitionInvocationRequest: ...
