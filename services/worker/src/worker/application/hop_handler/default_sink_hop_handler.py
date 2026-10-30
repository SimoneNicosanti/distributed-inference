from typing import override

from worker.application.hop_handler.abc.hop_handler import HopHandler
from worker.application.postbox.abc.outbox import Outbox
from worker.domain.hop.hop import SinkHop
from worker.domain.hop.hop_invocation import HopInvocationRequest, HopInvocationResult
from worker.domain.model.model_pass import ModelPassResult
from worker.domain.tensor.tensor_bundle import TensorBundle


class DefaultSinkHopHandler(HopHandler[SinkHop]):
    def __init__(self, outbox: Outbox[ModelPassResult]) -> None:
        super().__init__()
        self._outbox = outbox

    @override
    async def handle(
        self, hop_invocation_request: HopInvocationRequest[SinkHop]
    ) -> HopInvocationResult[SinkHop]:

        model_pass_result = ModelPassResult(
            context=hop_invocation_request.model_pass_context,
            bundle=hop_invocation_request.payload,
        )
        await self._outbox.deposit(model_pass_result)

        payload = TensorBundle({})
        return HopInvocationResult(
            context=hop_invocation_request.context, payload=payload
        )
