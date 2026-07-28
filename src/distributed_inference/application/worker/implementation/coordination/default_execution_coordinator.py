from typing import override

from distributed_inference.application.worker.contracts.execution.flow.execution_coordinator import (
    ExecutionCoordinator,
)
from distributed_inference.application.worker.contracts.execution.forwarding.inference_message_forwarder import (
    InferenceMessageForwarder,
)
from distributed_inference.application.worker.contracts.execution.forwarding.inference_message_gatherer import (
    InferenceMessageGatherer,
)
from distributed_inference.application.worker.contracts.execution.forwarding.inference_response_router import (
    InferenceResponseRouter,
)
from distributed_inference.application.worker.contracts.execution.inference.inference_coordinator import (
    InferenceCoordinator,
)
from distributed_inference.application.worker.domain.inference_flow import (
    InferenceMessage,
)


class DefaultExecutionCoordinator(ExecutionCoordinator):
    def __init__(
        self,
        inference_message_gatherer: InferenceMessageGatherer,
        inference_coordinator: InferenceCoordinator,
        inference_response_router: InferenceResponseRouter,
        inference_message_forwarder: InferenceMessageForwarder,
    ):
        self._inference_message_gatherer = inference_message_gatherer
        self._inference_coordinator = inference_coordinator
        self._inference_response_router = inference_response_router
        self._inference_message_forwarder = inference_message_forwarder

    @override
    async def process_inference_message(
        self, inference_message: InferenceMessage
    ) -> None:

        inference_request = (
            await self._inference_message_gatherer.gather_inference_message(
                inference_message
            )
        )
        if inference_request is None:
            ## Inference request could not be gathered
            ## Waiting
            return None

        inference_response = (
            await self._inference_coordinator.process_inference_request(
                inference_request
            )
        )

        route_instructions = (
            await self._inference_response_router.route_inference_response(
                inference_response
            )
        )

        for route_instruction in route_instructions:
            ## TODO: Build inference message for the next server
            self._inference_message_forwarder.forward_inference_message(
                inference_message, route_instruction
            )

        return None
