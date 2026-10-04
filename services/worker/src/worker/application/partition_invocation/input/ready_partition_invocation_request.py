import asyncio

from pydantic import BaseModel

from worker.domain.partition.partition_invocation import PartitionInvocationRequest
from worker.domain.partition.partition_invocation_contribution import (
    PartitionInvocationContribution,
)


class ReadyPartitionInvocationRequest(BaseModel):
    request: PartitionInvocationRequest
    contributions: list[PartitionInvocationContribution]
    future: asyncio.Future[None]
