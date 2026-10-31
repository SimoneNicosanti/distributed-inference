from typing import override

from shared.plan.hop_logical_id import IngressReplicaId
from worker.application.hop_handler.abc.hop_handler import HopHandler
from worker.domain.hop.hop_invocation import HopInvocationRequest, HopInvocationResult


class DefaultIngressHopHandler(HopHandler[IngressReplicaId]):
    @override
    async def handle(
        self, hop_invocation_request: HopInvocationRequest[IngressReplicaId]
    ) -> HopInvocationResult[IngressReplicaId]:

        return HopInvocationResult(
            context=hop_invocation_request.context,
            payload=hop_invocation_request.payload,
        )
