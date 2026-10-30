from abc import ABC, abstractmethod

from worker.domain.hop.hop import InvocableHop
from worker.domain.hop.hop_invocation import HopInvocationRequest, HopInvocationResult


## This is the inference coordinator for a pool of inference workers
class HopHandler[HopT: InvocableHop](ABC):
    @abstractmethod
    async def handle(
        self, hop_invocation_request: HopInvocationRequest[HopT]
    ) -> HopInvocationResult[HopT]: ...
