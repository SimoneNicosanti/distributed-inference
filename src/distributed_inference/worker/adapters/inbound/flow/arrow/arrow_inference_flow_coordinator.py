from typing import override

from distributed_inference.worker.application.ports.inbound.inference_flow_coordinator import (
    InferenceFlowCoordinator,
)
from distributed_inference.worker.domain.sub_model.invocation.sub_model_invocation_message import (
    SubModelInvocationAck,
    SubModelInvocationMessage,
)


class ArrowInferenceFlowCoordinator(InferenceFlowCoordinator):
    @override
    async def process_sub_model_invocation_message(
        self, sub_model_invocation_message: SubModelInvocationMessage
    ) -> SubModelInvocationAck:
        pass
