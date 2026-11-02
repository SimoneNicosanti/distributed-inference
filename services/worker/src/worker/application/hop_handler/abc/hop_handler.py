from abc import ABC, abstractmethod

from shared.plan.hop import PlannedInvocableHopId
from worker.domain.hop.hop_invocation import HopInvocationRequest, HopInvocationResult


## This is the inference coordinator for a pool of inference workers
class HopHandler[HopT: PlannedInvocableHopId](ABC):
    @abstractmethod
    async def handle(
        self, hop_invocation_request: HopInvocationRequest[HopT]
    ) -> HopInvocationResult[HopT]: ...
