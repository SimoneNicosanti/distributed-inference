from typing import override

from worker.application.hop_handler.abc.hop_handler import HopHandler
from worker.domain.hop.hop import IngressHop
from worker.domain.hop.hop_invocation import HopInvocationRequest, HopInvocationResult


class DefaultIngressHopHandler(HopHandler[IngressHop]):
    @override
    async def handle(
        self, hop_invocation_request: HopInvocationRequest[IngressHop]
    ) -> HopInvocationResult[IngressHop]:

        return HopInvocationResult(
            context=hop_invocation_request.context,
            payload=hop_invocation_request.payload,
        )
